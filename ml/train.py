"""Train, select and save the HydraSense groundwater-level forecasting model.

Run from the repository root:

    backend/.venv/Scripts/python.exe ml/train.py

Method (SRS 6.2, 6.3, 9.7)
-------------------------
* **Candidate models** - a persistence baseline (carry the origin level
  forward), a RandomForest and a Ridge regression. Each learned candidate is
  wrapped in a Pipeline and predicts the *change* from the origin level, because
  the level here trends strongly within the year: held-out levels fall outside
  the range of training levels, and a tree model cannot extrapolate to them.
* **Selection data** - a rolling-origin backtest: six 30-day evaluation windows
  spread over the year, each trained only on data whose target date precedes the
  window (expanding window, no shuffling). The most recent window is weighted
  most heavily, because forecasts are issued from the end of the series.
* **Hold-out** - the final 60 days are never used for selection, fitting order or
  tuning. They are scored once for reporting, so the reported error is a genuine
  held-out result.
* **Served configuration** - per horizon, a learned candidate is served only if
  it beats persistence on the recency-weighted backtest by at least
  ``SELECTION_MARGIN``. Otherwise persistence is served. The margin keeps the
  simpler, measured-better model unless the improvement is clear.
* **Served pipelines** - after selection, the chosen candidate is refit on every
  available origin (all data is legitimately available at forecast time).

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
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_preprocessing as dp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "DWLR_Dataset_2023.csv"
DEFAULT_MODEL = ROOT / "ml" / "model.pkl"

RANDOM_STATE = 42
MODEL_PARAMS = {
    "n_estimators": 300,
    "max_depth": 8,
    "min_samples_leaf": 3,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}
RIDGE_ALPHA = 1.0

KIND_PERSISTENCE = "persistence"
KIND_FOREST = "random_forest_delta"
KIND_RIDGE = "ridge_delta"
LEARNED_KINDS = (KIND_FOREST, KIND_RIDGE)
CANDIDATES = (KIND_PERSISTENCE,) + LEARNED_KINDS

BACKTEST_START_DAY = 100  # first backtest window starts this many days into the series
BACKTEST_FOLDS = 6
BACKTEST_FOLD_DAYS = 30
HOLDOUT_DAYS = 60
SELECTION_MARGIN = 0.25  # learned candidate must beat persistence by this fraction
DO_FILL_TRAIN_FRACTION = 0.70


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


def build_estimator(kind: str):
    """Fresh, unfitted estimator for a learned candidate."""
    if kind == KIND_FOREST:
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", RandomForestRegressor(**MODEL_PARAMS)),
            ]
        )
    if kind == KIND_RIDGE:
        return Pipeline(
            steps=[("scaler", StandardScaler()), ("model", Ridge(alpha=RIDGE_ALPHA))]
        )
    raise ValueError(f"{kind!r} is not a learned candidate")


def backtest_windows(frame: pd.DataFrame, holdout_start: pd.Timestamp) -> list[tuple]:
    """Rolling-origin evaluation windows, all strictly before the hold-out."""
    start = pd.Timestamp(frame[dp.DATE_COLUMN].min()).normalize()
    windows = []
    for index in range(BACKTEST_FOLDS):
        window_start = start + pd.Timedelta(days=BACKTEST_START_DAY + BACKTEST_FOLD_DAYS * index)
        window_end = window_start + pd.Timedelta(days=BACKTEST_FOLD_DAYS - 1)
        if window_end >= holdout_start:
            break
        windows.append((window_start, window_end))
    return windows


def recency_weights(n: int) -> np.ndarray:
    """Most recent window weighted highest (1, 1/2, 1/3, ...)."""
    weights = np.array([1.0 / (n - i) for i in range(n)], dtype=float)
    return weights / weights.sum()


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
    weights = recency_weights(len(windows))

    served_models: dict[int, dict] = {}
    metrics: dict[str, dict] = {}

    for horizon in range(1, horizon_days + 1):
        data = dp.build_supervised(
            frame, horizon, do_fill_value=do_fill, target_mode=dp.TARGET_MODE_DELTA
        )
        target_dates = pd.to_datetime(data.target_dates)
        actual = dp.to_levels(data.origin_levels, data.y.to_numpy(), data.target_mode)
        persistence_all = data.origin_levels.to_numpy(dtype=float)

        # --- rolling-origin backtest (selection uses only this) -----------------
        fold_scores: dict[str, list[float]] = {kind: [] for kind in CANDIDATES}
        for window_start, window_end in windows:
            train_idx = np.where(target_dates < window_start)[0]
            eval_idx = np.where(
                (target_dates >= window_start) & (target_dates <= window_end)
            )[0]
            if len(eval_idx) == 0 or len(train_idx) == 0:
                continue
            fold_scores[KIND_PERSISTENCE].append(
                dp.regression_metrics(actual[eval_idx], persistence_all[eval_idx])["mae"]
            )
            for kind in LEARNED_KINDS:
                estimator = build_estimator(kind)
                estimator.fit(data.X.iloc[train_idx], data.y.iloc[train_idx])
                predicted = dp.to_levels(
                    data.origin_levels, estimator.predict(data.X), data.target_mode
                )
                fold_scores[kind].append(
                    dp.regression_metrics(actual[eval_idx], predicted[eval_idx])["mae"]
                )

        backtest = {
            kind: float(np.average(fold_scores[kind], weights=weights))
            if fold_scores[kind]
            else float("nan")
            for kind in CANDIDATES
        }

        # --- read-only hold-out (reporting only, never selection) ---------------
        holdout_idx = np.where(target_dates >= holdout_start)[0]
        pre_holdout_idx = np.where(target_dates < holdout_start)[0]
        holdout: dict[str, float | int] = {
            "origins": int(len(holdout_idx)),
            "from": str(target_dates.iloc[holdout_idx[0]].date()),
            "to": str(target_dates.iloc[holdout_idx[-1]].date()),
            KIND_PERSISTENCE: dp.regression_metrics(
                actual[holdout_idx], persistence_all[holdout_idx]
            )["mae"],
        }
        for kind in LEARNED_KINDS:
            estimator = build_estimator(kind)
            estimator.fit(data.X.iloc[pre_holdout_idx], data.y.iloc[pre_holdout_idx])
            predicted = dp.to_levels(
                data.origin_levels, estimator.predict(data.X), data.target_mode
            )
            holdout[kind] = dp.regression_metrics(actual[holdout_idx], predicted[holdout_idx])[
                "mae"
            ]

        # --- selection (margin favouring persistence) ---------------------------
        best_kind = min(backtest, key=lambda kind: backtest[kind])
        if best_kind != KIND_PERSISTENCE and backtest[best_kind] > backtest[
            KIND_PERSISTENCE
        ] * (1 - SELECTION_MARGIN):
            best_kind = KIND_PERSISTENCE

        # --- refit the selected candidate on every available origin --------------
        if best_kind == KIND_PERSISTENCE:
            served_models[horizon] = {"kind": KIND_PERSISTENCE, "estimator": None}
        else:
            estimator = build_estimator(best_kind)
            estimator.fit(data.X, data.y)
            served_models[horizon] = {"kind": best_kind, "estimator": estimator}

        metrics[str(horizon)] = {
            "selected": best_kind,
            "backtest": {k: float(v) for k, v in backtest.items()},
            "holdout": holdout,
            "selected_holdout_mae": float(holdout[best_kind]),
            "selected_backtest_mae": float(backtest[best_kind]),
            "n_train_origins": int(len(pre_holdout_idx)),
        }

    horizons = list(range(1, horizon_days + 1))
    selected_holdout = [metrics[str(h)]["selected_holdout_mae"] for h in horizons]
    baseline_holdout = [metrics[str(h)][ "holdout"][KIND_PERSISTENCE] for h in horizons]
    forest_holdout = [metrics[str(h)]["holdout"][KIND_FOREST] for h in horizons]
    selected_backtest = [metrics[str(h)]["selected_backtest_mae"] for h in horizons]
    baseline_backtest = [metrics[str(h)]["backtest"][KIND_PERSISTENCE] for h in horizons]
    learned_selected = [h for h in horizons if metrics[str(h)]["selected"] != KIND_PERSISTENCE]

    artifact = {
        "artifact_format": 1,
        "model_type": "per-horizon selection: persistence | RandomForest(delta) | Ridge(delta)",
        "target": dp.TARGET_COLUMN,
        "target_mode": dp.TARGET_MODE_DELTA,
        "target_definition": (
            "Water level `horizon_days` days after the forecast origin (the last observed "
            "day). Learned candidates predict the change from the origin level; the served "
            "level is origin_level + predicted_change."
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
        "model_params": {"random_forest": MODEL_PARAMS, "ridge_alpha": RIDGE_ALPHA},
        "selection": {
            "rule": (
                "Per horizon: serve the learned candidate with the lowest recency-weighted "
                "rolling-origin backtest MAE only if it improves on persistence by at least "
                f"{SELECTION_MARGIN:.0%}; otherwise serve persistence."
            ),
            "margin": SELECTION_MARGIN,
            "backtest": {
                "windows": [[str(w[0].date()), str(w[1].date())] for w in windows],
                "fold_days": BACKTEST_FOLD_DAYS,
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
                    "selection, hyper-parameters or feature choice."
                ),
            },
        },
        "served": {str(h): served_models[h]["kind"] for h in horizons},
        "models": served_models,
        "metrics": {
            "by_horizon": metrics,
            "overall": {
                "selected_backtest_mae_mean": float(np.mean(selected_backtest)),
                "persistence_backtest_mae_mean": float(np.mean(baseline_backtest)),
                "selected_holdout_mae_mean": float(np.mean(selected_holdout)),
                "persistence_holdout_mae_mean": float(np.mean(baseline_holdout)),
                "forest_holdout_mae_mean": float(np.mean(forest_holdout)),
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
          f"[{artifact['target_mode']} mode for learned candidates]")
    print(f"horizon            : 1..{artifact['horizon_days']} days (direct multi-step)")
    print(f"features           : {len(artifact['feature_order'])}")
    print(f"backtest windows   : {artifact['selection']['backtest']['windows']}")
    print(f"hold-out (read-only): {artifact['selection']['holdout']['from']} -> "
          f"{artifact['selection']['holdout']['to']}")
    print()
    print(f"{'h':>3} {'served':>12} {'bt base':>9} {'bt forest':>10} {'bt ridge':>9} "
          f"{'ho base':>9} {'ho forest':>10} {'ho ridge':>9}")
    print("-" * 96)
    for horizon in sorted(int(h) for h in artifact["metrics"]["by_horizon"]):
        row = artifact["metrics"]["by_horizon"][str(horizon)]
        bt = row["backtest"]
        ho = row["holdout"]
        print(
            f"{horizon:>3} {row['selected']:>12} {bt[KIND_PERSISTENCE]:>9.4f} "
            f"{bt[KIND_FOREST]:>10.4f} {bt[KIND_RIDGE]:>9.4f} "
            f"{ho[KIND_PERSISTENCE]:>9.4f} {ho[KIND_FOREST]:>10.4f} {ho[KIND_RIDGE]:>9.4f}"
        )
    print("-" * 96)
    print("bt = recency-weighted rolling-origin backtest (used for selection)")
    print("ho = held-out final window (reporting only, never used for selection)")
    print()
    print(f"selected backtest MAE mean {overall['selected_backtest_mae_mean']:.4f} m  "
          f"vs persistence {overall['persistence_backtest_mae_mean']:.4f} m")
    print(f"selected hold-out MAE mean {overall['selected_holdout_mae_mean']:.4f} m  "
          f"vs persistence {overall['persistence_holdout_mae_mean']:.4f} m  "
          f"vs forest {overall['forest_holdout_mae_mean']:.4f} m")
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
        help="forecast horizon in days; must stay supported by the backtest and hold-out",
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
