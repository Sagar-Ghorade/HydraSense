"""Train, select and save the HydraSense groundwater-level forecasting model.

Run from the repository root:

    backend/.venv/Scripts/python.exe ml/train.py

Method (SRS 6.2, 6.3, 9.7)
-------------------------
* **Candidates** - a persistence baseline (carry the origin level forward) and a
  Ridge regression predicting the *change* from the origin level, standardised
  through a Pipeline. Every feature is known at the forecast origin; the only
  advance information is the target date's day-of-year (see
  :mod:`data_preprocessing`).
* **Why a regularised linear model** - the series is one year of daily data
  (~335 usable origins per horizon) with strongly collinear level/lag features.
  A high-variance learner such as a forest fits the one-off monsoon jump well
  and then predicts 3-4x the real error in flatter regimes, so the candidate set
  is deliberately small and the penalty strength is selected rather than
  guessed. The slope features (`wl_slope7`, `wl_slope30`) give the linear model
  an explicit metres-per-day trend it can scale by the horizon, which is what
  lets it extrapolate where a tree cannot.
* **Selection data** - a rolling-origin backtest of six 30-day windows that are
  *anchored to the hold-out*: the newest window ends the day before the hold-out
  starts, so every fold is trained only on earlier data and the evaluation ends
  where the real forecast begins. The most recent window is weighted most
  heavily (1, 1/2, 1/3, ... over folds), because forecasts are issued from the
  end of the series.
* **Hyper-parameter selection** - the Ridge penalty is chosen per horizon from
  ``ALPHA_GRID`` on those same folds (cross-validation). Candidate and penalty
  choice therefore both use the backtest only.
* **Hold-out** - the final ``HOLDOUT_DAYS`` are never used for selection,
  tuning or fitting order. They are scored once, for reporting, so the reported
  error is a genuine held-out result.
* **Served configuration** - per horizon, Ridge is served only if it improves the
  recency-weighted backtest MAE on persistence by at least ``MIN_GAIN``;
  otherwise persistence is served. The selected candidate is then refit on every
  available origin (all data is legitimately available at forecast time).
* **Uncertainty** - per horizon the artifact stores empirical quantiles of the
  pooled backtest residuals of the *served* candidate, so the API can publish a
  prediction interval instead of a bare point estimate.

Every number printed comes from this run; nothing is hardcoded or estimated.
"""

from __future__ import annotations

import argparse
import hashlib
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_preprocessing as dp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "DWLR_Dataset_2023.csv"
DEFAULT_MODEL = ROOT / "ml" / "model.pkl"

KIND_PERSISTENCE = "persistence"
KIND_RIDGE = "ridge_delta"
LEARNED_KINDS = (KIND_RIDGE,)
CANDIDATES = (KIND_PERSISTENCE,) + LEARNED_KINDS

#: Penalty grid cross-validated per horizon. The top of the grid matters: as
#: alpha grows Ridge shrinks towards a constant per-day drift, which is a real
#: (and often the best) model for this series rather than a degenerate one.
ALPHA_GRID: tuple[float, ...] = (
    10.0,
    30.0,
    100.0,
    300.0,
    1_000.0,
    3_000.0,
    10_000.0,
    30_000.0,
    100_000.0,
    300_000.0,
    1_000_000.0,
    3_000_000.0,
)

BACKTEST_FOLDS = 6
BACKTEST_FOLD_DAYS = 30
HOLDOUT_DAYS = 60
#: A learned candidate must reduce the backtest MAE by at least this fraction.
MIN_GAIN = 0.05
DO_FILL_TRAIN_FRACTION = 0.70
MIN_HOLDOUT_TRAIN_ORIGINS = 60
RESIDUAL_QUANTILES = (0.10, 0.50, 0.90)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def build_estimator(alpha: float):
    """Fresh, unfitted Ridge pipeline (standardised features, delta target)."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=float(alpha), random_state=None)),
        ]
    )


def backtest_windows(frame: pd.DataFrame, holdout_start: pd.Timestamp) -> list[tuple]:
    """Rolling-origin evaluation windows ending the day before the hold-out.

    Anchoring on the hold-out rather than on a fixed day count keeps the folds
    representative of the regime the model is actually used in: forecasts are
    issued from the end of the series, so folds that stop months earlier only
    measure how well the model describes the middle of the year.
    """
    holdout_start = pd.Timestamp(holdout_start).normalize()
    windows: list[tuple] = []
    for index in range(BACKTEST_FOLDS):
        window_end = holdout_start - pd.Timedelta(days=1 + BACKTEST_FOLD_DAYS * index)
        window_start = window_end - pd.Timedelta(days=BACKTEST_FOLD_DAYS - 1)
        windows.append((window_start, window_end))
    return sorted(windows)


def recency_weights(n: int) -> np.ndarray:
    """Most recent window weighted highest (1, 1/2, 1/3, ...)."""
    if n < 1:
        raise ValueError("recency_weights needs at least one fold")
    weights = np.array([1.0 / (n - i) for i in range(n)], dtype=float)
    return weights / weights.sum()


def fold_indices(target_dates: pd.Series, windows: list[tuple]) -> list[tuple]:
    """(train, eval) row indices per window, dropping empty windows.

    Training rows are those whose *target* precedes the window start, so no
    evaluation row was ever seen while fitting.
    """
    target_dates = pd.to_datetime(target_dates)
    folds = []
    for window_start, window_end in windows:
        train_idx = np.where(target_dates < pd.Timestamp(window_start))[0]
        eval_idx = np.where(
            (target_dates >= pd.Timestamp(window_start))
            & (target_dates <= pd.Timestamp(window_end))
        )[0]
        if len(eval_idx) == 0 or len(train_idx) == 0:
            continue
        folds.append((train_idx, eval_idx))
    return folds


def evaluate_backtest(
    data: dp.Supervised,
    windows: list[tuple],
    alphas: tuple[float, ...] = ALPHA_GRID,
) -> dict:
    """Score persistence and every Ridge penalty on the rolling folds.

    Returns fold MAEs, recency-weighted MAEs, the best alpha and a prediction
    array per candidate - exactly what the selection rule and the uncertainty
    quantiles need. Shared by ``train.py`` and ``evaluate.py`` so the recorded
    numbers and the reproduced numbers cannot disagree.
    """
    actual = dp.to_levels(data.origin_levels, data.y.to_numpy(), data.target_mode)
    origin_levels = data.origin_levels.to_numpy(dtype=float)
    folds = fold_indices(data.target_dates, windows)
    if not folds:
        raise ValueError("No usable backtest window: every fold was empty")
    weights = recency_weights(len(folds))

    persistence_fold: list[float] = []
    ridge_fold: dict[float, list[float]] = {float(a): [] for a in alphas}
    ridge_prediction: dict[float, np.ndarray] = {
        float(a): np.full(len(actual), np.nan) for a in alphas
    }

    for train_idx, eval_idx in folds:
        persistence_fold.append(
            dp.regression_metrics(actual[eval_idx], origin_levels[eval_idx])["mae"]
        )
        for alpha in alphas:
            estimator = build_estimator(alpha)
            estimator.fit(data.X.iloc[train_idx], data.y.iloc[train_idx])
            predicted = dp.to_levels(
                origin_levels[eval_idx],
                estimator.predict(data.X.iloc[eval_idx]),
                data.target_mode,
            )
            ridge_fold[float(alpha)].append(
                dp.regression_metrics(actual[eval_idx], predicted)["mae"]
            )
            ridge_prediction[float(alpha)][eval_idx] = predicted

    weighted_persistence = float(np.average(persistence_fold, weights=weights))
    weighted_ridge = {
        alpha: float(np.average(scores, weights=weights))
        for alpha, scores in ridge_fold.items()
    }
    best_alpha = min(weighted_ridge, key=lambda alpha: weighted_ridge[alpha])
    return {
        "folds": folds,
        "weights": weights,
        "actual": actual,
        "origin_levels": origin_levels,
        "persistence_fold": persistence_fold,
        "ridge_fold": ridge_fold,
        "ridge_prediction": ridge_prediction,
        "weighted_persistence": weighted_persistence,
        "weighted_ridge": weighted_ridge,
        "best_alpha": best_alpha,
        "best_ridge_mae": weighted_ridge[best_alpha],
    }


def select_kind(weighted_persistence: float, weighted_learned: float) -> str:
    """Serve the learned candidate only if it clears the gain threshold."""
    if weighted_learned <= weighted_persistence * (1.0 - MIN_GAIN):
        return KIND_RIDGE
    return KIND_PERSISTENCE


def evaluate_holdout(data: dp.Supervised, holdout_start: pd.Timestamp, alpha: float) -> dict:
    """Score persistence and the chosen Ridge on the untouched final window."""
    target_dates = pd.to_datetime(data.target_dates)
    actual = dp.to_levels(data.origin_levels, data.y.to_numpy(), data.target_mode)
    origin_levels = data.origin_levels.to_numpy(dtype=float)

    holdout_idx = np.where(target_dates >= pd.Timestamp(holdout_start))[0]
    pre_idx = np.where(target_dates < pd.Timestamp(holdout_start))[0]
    if len(holdout_idx) == 0:
        raise ValueError("No hold-out rows for this horizon")
    if len(pre_idx) < MIN_HOLDOUT_TRAIN_ORIGINS:
        raise ValueError(
            f"Only {len(pre_idx)} origins precede the hold-out; "
            f"{MIN_HOLDOUT_TRAIN_ORIGINS} are required"
        )

    estimator = build_estimator(alpha)
    estimator.fit(data.X.iloc[pre_idx], data.y.iloc[pre_idx])
    predicted = dp.to_levels(
        origin_levels[holdout_idx],
        estimator.predict(data.X.iloc[holdout_idx]),
        data.target_mode,
    )
    return {
        "origins": int(len(holdout_idx)),
        "from": str(pd.Timestamp(target_dates.iloc[holdout_idx[0]]).date()),
        "to": str(pd.Timestamp(target_dates.iloc[holdout_idx[-1]]).date()),
        "n_train_origins": int(len(pre_idx)),
        KIND_PERSISTENCE: dp.regression_metrics(
            actual[holdout_idx], origin_levels[holdout_idx]
        ),
        KIND_RIDGE: dp.regression_metrics(actual[holdout_idx], predicted),
    }


def evaluate_horizon(
    frame: pd.DataFrame,
    horizon: int,
    windows: list[tuple],
    holdout_start: pd.Timestamp,
    do_fill: float,
    alphas: tuple[float, ...] = ALPHA_GRID,
) -> dict:
    """Full evaluation for one horizon: backtest, selection, hold-out, residuals.

    ``train.py`` uses it to record metrics; ``ml/evaluate.py`` uses it to
    reproduce them from scratch.
    """
    data = dp.build_supervised(
        frame, horizon, do_fill_value=do_fill, target_mode=dp.TARGET_MODE_DELTA
    )
    backtest = evaluate_backtest(data, windows, alphas=alphas)
    selected = select_kind(
        backtest["weighted_persistence"], backtest["best_ridge_mae"]
    )
    holdout = evaluate_holdout(data, holdout_start, backtest["best_alpha"])

    if selected == KIND_RIDGE:
        residuals = (
            backtest["ridge_prediction"][backtest["best_alpha"]] - backtest["actual"]
        )
        residuals = residuals[np.isfinite(residuals)]
    else:
        residuals = backtest["origin_levels"] - backtest["actual"]

    quantiles = np.quantile(residuals, RESIDUAL_QUANTILES).tolist()
    residual_quantiles = {
        f"p{int(round(q * 100))}": float(value)
        for q, value in zip(RESIDUAL_QUANTILES, quantiles)
    }
    return {
        "data": data,
        "backtest": backtest,
        "selected": selected,
        "alpha": backtest["best_alpha"],
        "holdout": holdout,
        "residuals": residuals,
        "residual_quantiles": residual_quantiles,
    }


def train(dataset_path: Path, model_path: Path, horizon_days: int) -> dict:
    frame, report = dp.clean(dp.load_raw(dataset_path))
    start = pd.Timestamp(frame[dp.DATE_COLUMN].min()).normalize()
    end = pd.Timestamp(frame[dp.DATE_COLUMN].max()).normalize()

    if len(frame) < dp.MIN_HISTORY_DAYS * 3:
        raise SystemExit(
            f"Dataset has only {len(frame)} usable rows; at least "
            f"{dp.MIN_HISTORY_DAYS * 3} are needed for the evaluation design."
        )

    # Training-only statistic for the dissolved-oxygen fill value.
    do_fill = dp.train_do_fill_value(
        frame, np.arange(0, int(len(frame) * DO_FILL_TRAIN_FRACTION))
    )

    holdout_start = end - pd.Timedelta(days=HOLDOUT_DAYS - 1)
    windows = backtest_windows(frame, holdout_start)

    served_models: dict[int, dict] = {}
    metrics: dict[str, dict] = {}

    for horizon in range(1, horizon_days + 1):
        result = evaluate_horizon(frame, horizon, windows, holdout_start, do_fill)
        backtest = result["backtest"]
        holdout = result["holdout"]
        selected = result["selected"]
        alpha = result["alpha"]

        if selected == KIND_PERSISTENCE:
            served_models[horizon] = {"kind": KIND_PERSISTENCE, "estimator": None}
        else:
            estimator = build_estimator(alpha)
            estimator.fit(result["data"].X, result["data"].y)
            served_models[horizon] = {"kind": KIND_RIDGE, "estimator": estimator}

        metrics[str(horizon)] = {
            "selected": selected,
            "alpha": alpha if selected == KIND_RIDGE else None,
            "backtest": {
                KIND_PERSISTENCE: backtest["weighted_persistence"],
                KIND_RIDGE: backtest["best_ridge_mae"],
            },
            "backtest_by_alpha": {
                str(alpha): score for alpha, score in backtest["weighted_ridge"].items()
            },
            "backtest_folds": {
                KIND_PERSISTENCE: [float(v) for v in backtest["persistence_fold"]],
                KIND_RIDGE: [
                    float(v) for v in backtest["ridge_fold"][backtest["best_alpha"]]
                ],
            },
            "holdout": holdout,
            "selected_holdout_mae": float(holdout[selected]["mae"]),
            "selected_backtest_mae": float(
                backtest["weighted_persistence"]
                if selected == KIND_PERSISTENCE
                else backtest["best_ridge_mae"]
            ),
            "residual_mae": float(np.mean(np.abs(result["residuals"]))),
            "residual_quantiles": result["residual_quantiles"],
            "n_train_origins": int(holdout["n_train_origins"]),
            "n_backtest_folds": len(backtest["folds"]),
        }

    horizons = list(range(1, horizon_days + 1))
    selected_holdout = [metrics[str(h)]["selected_holdout_mae"] for h in horizons]
    baseline_holdout = [metrics[str(h)]["holdout"][KIND_PERSISTENCE]["mae"] for h in horizons]
    selected_backtest = [metrics[str(h)]["selected_backtest_mae"] for h in horizons]
    baseline_backtest = [metrics[str(h)]["backtest"][KIND_PERSISTENCE] for h in horizons]
    ridge_backtest = [metrics[str(h)]["backtest"][KIND_RIDGE] for h in horizons]
    learned_selected = [h for h in horizons if metrics[str(h)]["selected"] != KIND_PERSISTENCE]

    artifact = {
        "artifact_format": 2,
        "model_type": "per-horizon selection: persistence | Ridge(delta, standardised)",
        "target": dp.TARGET_COLUMN,
        "target_mode": dp.TARGET_MODE_DELTA,
        "target_definition": (
            "Water level `horizon_days` days after the forecast origin (the last observed "
            "day). The learned candidate predicts the change from the origin level; the "
            "served level is origin_level + predicted_change."
        ),
        "unit": dp.TARGET_UNIT,
        "unit_confirmed": dp.UNIT_CONFIRMED,
        "unit_note": (
            "Unit is taken from the dataset column name (Water_Level_m). The depth datum is "
            "not documented in the source file, so 'below ground level' is a label only and "
            "is not independently verified."
        ),
        "horizon_days": horizon_days,
        "feature_order": list(dp.FEATURE_ORDER),
        "do_fill_value": do_fill,
        "model_params": {
            "ridge_alpha_grid": list(ALPHA_GRID),
            "alpha_selection": "chosen per horizon on the backtest folds (cross-validation)",
            "min_gain": MIN_GAIN,
        },
        "candidates": list(CANDIDATES),
        "selection": {
            "rule": (
                "Per horizon: serve Ridge only if its recency-weighted rolling-origin "
                f"backtest MAE is at least {MIN_GAIN:.0%} below persistence; otherwise "
                "serve persistence. The Ridge penalty is chosen per horizon on the same "
                "folds, so selection never touches the hold-out."
            ),
            "min_gain": MIN_GAIN,
            "backtest": {
                "windows": [[str(w[0].date()), str(w[1].date())] for w in windows],
                "fold_days": BACKTEST_FOLD_DAYS,
                "folds": BACKTEST_FOLDS,
                "anchored_to": "last window ends the day before the hold-out starts",
                "weighting": "recency-weighted (most recent window weighted highest)",
                "expanding_window": True,
            },
            "holdout": {
                "days": HOLDOUT_DAYS,
                "from": str(holdout_start.date()),
                "to": str(end.date()),
                "used_for_selection": False,
                "note": (
                    "The hold-out window is scored for reporting only. It never influenced "
                    "candidate choice, alpha, selection or fitting order."
                ),
            },
        },
        "uncertainty": {
            "method": (
                "Empirical quantiles of the pooled rolling-backtest residuals "
                "(predicted - actual) of the served candidate, per horizon."
            ),
            "quantiles": [f"p{int(round(q * 100))}" for q in RESIDUAL_QUANTILES],
            "interval": "p10..p90 = an 80% empirical band around each forecast point",
        },
        "served": {str(h): served_models[h]["kind"] for h in horizons},
        "models": served_models,
        "metrics": {
            "by_horizon": metrics,
            "overall": {
                "selected_backtest_mae_mean": float(np.mean(selected_backtest)),
                "persistence_backtest_mae_mean": float(np.mean(baseline_backtest)),
                "ridge_backtest_mae_mean": float(np.mean(ridge_backtest)),
                "selected_holdout_mae_mean": float(np.mean(selected_holdout)),
                "persistence_holdout_mae_mean": float(np.mean(baseline_holdout)),
                "learned_horizons": learned_selected,
                "persistence_horizons": [h for h in horizons if h not in learned_selected],
                "served_beats_persistence_on_backtest": bool(
                    np.mean(selected_backtest) < np.mean(baseline_backtest)
                ),
                "served_beats_persistence_on_holdout": bool(
                    np.mean(selected_holdout) < np.mean(baseline_holdout)
                ),
                "evaluated_on": "chronological rolling-origin backtest and a read-only hold-out",
            },
            "baseline": "persistence (carry the origin level forward)",
        },
        "dataset": {
            "path": _relative(dataset_path),
            "sha256": sha256_of(dataset_path),
            "rows": int(len(frame)),
            "start": str(start.date()),
            "end": str(end.date()),
            "target_column": dp.TARGET_COLUMN,
        },
        "cleaning": report.as_dict(),
        "library_versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path, compress=3)
    return artifact


def print_report(artifact: dict, model_path: Path) -> None:
    overall = artifact["metrics"]["overall"]
    print("HydraSense model training")
    print("=" * 96)
    print(f"dataset            : {artifact['dataset']['path']}  "
          f"({artifact['dataset']['rows']} rows, {artifact['dataset']['start']} -> "
          f"{artifact['dataset']['end']})")
    print(f"target             : {artifact['target']} ({artifact['unit']}) "
          f"[{artifact['target_mode']} mode for the learned candidate]")
    print(f"horizon            : 1..{artifact['horizon_days']} days (direct multi-step)")
    print(f"features           : {len(artifact['feature_order'])}")
    print(f"candidates         : {', '.join(artifact['candidates'])}")
    print(f"alpha grid         : {artifact['model_params']['ridge_alpha_grid']}")
    print(f"backtest windows   : {artifact['selection']['backtest']['windows']}")
    print(f"hold-out (read-only): {artifact['selection']['holdout']['from']} -> "
          f"{artifact['selection']['holdout']['to']}")
    print()
    print(f"{'h':>3} {'served':>12} {'alpha':>9} {'bt base':>9} {'bt ridge':>9} "
          f"{'ho base':>9} {'ho ridge':>9} {'gain':>7}")
    print("-" * 96)
    for horizon in sorted(int(h) for h in artifact["metrics"]["by_horizon"]):
        row = artifact["metrics"]["by_horizon"][str(horizon)]
        bt = row["backtest"]
        ho = row["holdout"]
        gain = 1.0 - bt[KIND_RIDGE] / bt[KIND_PERSISTENCE]
        alpha = str(row["alpha"]) if row["alpha"] else "-"
        print(
            f"{horizon:>3} {row['selected']:>12} {alpha:>9} "
            f"{bt[KIND_PERSISTENCE]:>9.4f} {bt[KIND_RIDGE]:>9.4f} "
            f"{ho[KIND_PERSISTENCE]['mae']:>9.4f} {ho[KIND_RIDGE]['mae']:>9.4f} "
            f"{gain:>6.1%}"
        )
    print("-" * 96)
    print("bt = recency-weighted rolling-origin backtest (selection + alpha)")
    print("ho = held-out final window (reporting only, never used for selection)")
    print("gain = backtest MAE reduction of Ridge over persistence")
    print()
    print(f"Ridge backtest MAE mean {overall['ridge_backtest_mae_mean']:.4f} m  "
          f"vs persistence {overall['persistence_backtest_mae_mean']:.4f} m")
    print(f"served hold-out MAE mean {overall['selected_holdout_mae_mean']:.4f} m  "
          f"vs persistence {overall['persistence_holdout_mae_mean']:.4f} m")
    print(f"learned candidates served for horizons: {overall['learned_horizons'] or 'none'}")
    print(f"served beats persistence on backtest: "
          f"{overall['served_beats_persistence_on_backtest']}   "
          f"on hold-out: {overall['served_beats_persistence_on_holdout']}")
    print()
    print("These are measured results for this single dataset, not accuracy guarantees.")
    print(f"artifact           : {model_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the HydraSense forecasting model")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--horizon",
        type=int,
        default=30,
        help="forecast horizon in days; must stay supported by the hold-out",
    )
    args = parser.parse_args()

    if args.horizon < 1 or args.horizon > HOLDOUT_DAYS:
        raise SystemExit(
            f"--horizon must be between 1 and {HOLDOUT_DAYS} days so that the hold-out "
            "window can score every horizon"
        )

    artifact = train(args.dataset, args.model_out, args.horizon)
    print_report(artifact, args.model_out)


if __name__ == "__main__":
    main()
