"""Independently re-evaluate a saved HydraSense model artifact.

Run from the repository root (the server does not need to be stopped):

    backend/.venv/Scripts/python.exe ml/evaluate.py

The script reloads the artifact and the raw dataset and then recomputes

* the recency-weighted rolling-origin backtest for both candidates,
* the chosen Ridge penalty for every horizon,
* the selection decision, re-derived from the stored rule and gain threshold
  rather than trusting the stored ``selected`` label,
* the read-only hold-out score,
* the pooled residual quantiles published as uncertainty,
* the served ``kinds`` map against the per-horizon selections,
* one forecast through the same code path the API uses,

and fails (exit status 1) if any recorded number does not reproduce, if the
selection does not follow the documented rule, if a served model cannot be
loaded and applied, or if the artifact does not match the current feature code
or dataset. Nothing is accepted on trust (SRS 6.2, 9.7).

The recomputation lives in ``ml/train.py`` (``evaluate_horizon``), so training
and verification share one implementation and cannot drift apart; everything
checked here is a comparison against independently stored values.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_preprocessing as dp  # noqa: E402
import train as training  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "ml" / "model.pkl"
DEFAULT_DATASET = ROOT / "data" / "DWLR_Dataset_2023.csv"
TOLERANCE = 1e-9


def _close(left, right, tolerance: float = TOLERANCE) -> bool:
    try:
        return abs(float(left) - float(right)) < tolerance
    except (TypeError, ValueError):
        return False


def evaluate(model_path: Path, dataset_path: Path) -> int:
    failures: list[str] = []

    try:
        artifact = joblib.load(model_path)
    except Exception as error:  # noqa: BLE001 - reported as a failure below
        print(f"could not load artifact {model_path}: {error}")
        return 1

    try:
        frame, _ = dp.clean(dp.load_raw(dataset_path))
    except Exception as error:  # noqa: BLE001
        print(f"could not load dataset {dataset_path}: {error}")
        return 1

    end = pd.Timestamp(frame[dp.DATE_COLUMN].max()).normalize()
    do_fill = float(artifact["do_fill_value"])
    horizon_days = int(artifact["horizon_days"])
    holdout_start = end - pd.Timedelta(days=training.HOLDOUT_DAYS - 1)
    windows = training.backtest_windows(frame, holdout_start)
    min_gain = float(artifact["model_params"]["min_gain"])

    window_rows = [[str(w[0].date()), str(w[1].date())] for w in windows]

    print("HydraSense model evaluation")
    print("=" * 104)
    print(f"artifact      : {model_path}")
    print(f"trained at    : {artifact['trained_at']}")
    print(f"library       : trained with scikit-learn "
          f"{artifact['library_versions']['scikit-learn']}, reloaded with {sklearn.__version__}")
    print(f"horizon       : 1..{horizon_days} days")
    print(f"features      : {len(artifact['feature_order'])} "
          f"(current code expects {len(dp.FEATURE_ORDER)})")
    print(f"backtest      : {window_rows}")
    print(f"hold-out      : {holdout_start.date()} -> {end.date()} (read-only)")
    print(f"trained clean : {artifact['cleaning']['source_rows']} source rows, "
          f"{artifact['dataset']['rows']} usable")

    # --- artifact vs current code and dataset ----------------------------------
    if list(artifact["feature_order"]) != list(dp.FEATURE_ORDER):
        failures.append(
            "artifact feature_order != ml/data_preprocessing.FEATURE_ORDER; the backend "
            "would build mismatched feature rows"
        )
    if artifact["dataset"]["sha256"] != training.sha256_of(dataset_path):
        failures.append(
            "dataset hash differs from the one recorded at training time; metrics below "
            "describe a different dataset"
        )
    if tuple(artifact["model_params"]["ridge_alpha_grid"]) != training.ALPHA_GRID:
        failures.append(
            "artifact alpha grid != training.ALPHA_GRID; the stored alpha is no longer "
            "the one that would be selected"
        )
    if not _close(min_gain, training.MIN_GAIN):
        failures.append(
            f"artifact min_gain {min_gain} != training.MIN_GAIN {training.MIN_GAIN}"
        )
    if tuple(artifact["selection"]["backtest"]["windows"]) != tuple(window_rows):
        failures.append("artifact backtest windows differ from the current protocol")

    print()
    print(f"{'h':>3} {'served':>12} {'alpha':>9} {'bt base':>9} {'bt ridge':>9} "
          f"{'ho base':>9} {'ho ridge':>9} {'sel ok':>7} {'match':>6}")
    print("-" * 104)

    recomputed_selected: dict[str, str] = {}
    recomputed_backtest: dict[str, dict] = {}
    recomputed_holdout: dict[str, dict] = {}

    for horizon in range(1, horizon_days + 1):
        stored = artifact["metrics"]["by_horizon"][str(horizon)]
        try:
            result = training.evaluate_horizon(
                frame, horizon, windows, holdout_start, do_fill
            )
        except Exception as error:  # noqa: BLE001 - reported as a failure below
            failures.append(f"horizon {horizon}: recomputation failed: {error}")
            continue

        backtest = result["backtest"]
        holdout = result["holdout"]
        backtest_values = {
            training.KIND_PERSISTENCE: backtest["weighted_persistence"],
            training.KIND_RIDGE: backtest["best_ridge_mae"],
        }
        recomputed_selected[str(horizon)] = result["selected"]
        recomputed_backtest[str(horizon)] = dict(backtest_values)
        recomputed_holdout[str(horizon)] = holdout

        # Re-derive the selection from the stored rule instead of trusting it.
        threshold = backtest["weighted_persistence"] * (1.0 - min_gain)
        derived = (
            training.KIND_RIDGE if backtest["best_ridge_mae"] <= threshold
            else training.KIND_PERSISTENCE
        )
        selection_ok = derived == stored["selected"]

        numbers_match = (
            all(
                _close(backtest_values[kind], stored["backtest"][kind])
                for kind in training.CANDIDATES
            )
            and all(
                _close(holdout[kind]["mae"], stored["holdout"][kind]["mae"])
                for kind in training.CANDIDATES
            )
        )
        fold_match = all(
            _close(a, b)
            for kind in training.CANDIDATES
            for a, b in zip(
                {
                    training.KIND_PERSISTENCE: backtest["persistence_fold"],
                    training.KIND_RIDGE: backtest["ridge_fold"][backtest["best_alpha"]],
                }[kind],
                stored["backtest_folds"][kind],
            )
        )
        # For a persistence horizon no alpha is recorded; otherwise the stored one
        # must be the minimum of the stored grid search.
        alpha_match = True
        if stored["alpha"] is not None:
            alpha_match = _close(result["alpha"], stored["alpha"])
            grid = stored["backtest_by_alpha"]
            alpha_match = alpha_match and _close(
                backtest["best_alpha"], min(grid, key=lambda key: grid[key])
            )
        quantile_match = all(
            _close(result["residual_quantiles"][key], value)
            for key, value in stored["residual_quantiles"].items()
        )
        numbers_match = numbers_match and fold_match and alpha_match and quantile_match

        if not selection_ok:
            failures.append(
                f"horizon {horizon}: stored rule gives {derived!r} but the artifact "
                f"stores {stored['selected']!r}"
            )
        if not numbers_match:
            failures.append(f"horizon {horizon}: recomputed metrics differ from stored values")

        alpha = str(stored["alpha"]) if stored["alpha"] is not None else "-"
        print(
            f"{horizon:>3} {stored['selected']:>12} {alpha:>9} "
            f"{backtest_values[training.KIND_PERSISTENCE]:>9.4f} "
            f"{backtest_values[training.KIND_RIDGE]:>9.4f} "
            f"{holdout[training.KIND_PERSISTENCE]['mae']:>9.4f} "
            f"{holdout[training.KIND_RIDGE]['mae']:>9.4f} "
            f"{'yes' if selection_ok else 'NO':>7} "
            f"{'yes' if numbers_match else 'NO':>6}"
        )

    # --- overall metrics --------------------------------------------------------
    overall = artifact["metrics"]["overall"]
    horizons = [str(h) for h in range(1, horizon_days + 1)]
    expected_overall = {
        "selected_backtest_mae_mean": float(
            np.mean(
                [
                    recomputed_backtest[h][
                        recomputed_selected[h]
                    ]
                    for h in horizons
                ]
            )
        ),
        "persistence_backtest_mae_mean": float(
            np.mean(
                [recomputed_backtest[h][training.KIND_PERSISTENCE] for h in horizons]
            )
        ),
        "selected_holdout_mae_mean": float(
            np.mean(
                [
                    recomputed_holdout[h][recomputed_selected[h]]["mae"]
                    for h in horizons
                ]
            )
        ),
        "persistence_holdout_mae_mean": float(
            np.mean(
                [recomputed_holdout[h][training.KIND_PERSISTENCE]["mae"] for h in horizons]
            )
        ),
    }
    for key, value in expected_overall.items():
        if not _close(value, overall[key]):
            failures.append(
                f"overall metric {key!r}: recomputed {value} != stored {overall[key]}"
            )

    # --- served map vs per-horizon selection ------------------------------------
    served = artifact["served"]
    for key, expected in recomputed_selected.items():
        if served.get(key) != expected:
            failures.append(
                f"served[{key!r}] is {served.get(key)!r} but the selection rule "
                f"reproduces {expected!r}; the API would serve a model whose recorded "
                "metrics describe a different candidate"
            )
        entry = artifact["models"].get(key) or artifact["models"].get(int(key))
        if entry is None:
            failures.append(f"models[{key}] missing from the artifact")
        elif entry["kind"] != expected:
            failures.append(
                f"models[{key}].kind is {entry['kind']!r}, selection says {expected!r}"
            )
        elif expected != training.KIND_PERSISTENCE and entry["estimator"] is None:
            failures.append(f"models[{key}] is learned but has no fitted estimator")

    print("-" * 104)
    print(f"all stored metrics and selections reproduced: {not failures}")
    print(f"served backtest MAE mean "
          f"{expected_overall['selected_backtest_mae_mean']:.4f} m vs "
          f"persistence {expected_overall['persistence_backtest_mae_mean']:.4f} m")
    print(f"served hold-out MAE mean "
          f"{expected_overall['selected_holdout_mae_mean']:.4f} m vs "
          f"persistence {expected_overall['persistence_holdout_mae_mean']:.4f} m")
    print(f"learned candidates served for horizons: "
          f"{overall['learned_horizons'] or 'none'}")
    print(f"uncertainty: {artifact['uncertainty']['method']}")
    print(f"              {artifact['uncertainty']['interval']}")

    # --- inference smoke test: exactly what the API does -------------------------
    print()
    print("inference smoke test (forecast issued from the last observed day)")
    last_date = pd.Timestamp(frame[dp.DATE_COLUMN].iloc[-1])
    last_level = float(frame[dp.TARGET_COLUMN].iloc[-1])
    print(f"last observation: {last_date.date()} -> {last_level:.3f} {artifact['unit']}")
    try:
        for horizon in (1, 7, max(1, int(horizon_days / 2)), horizon_days):
            row = dp.inference_row(frame, horizon, do_fill_value=do_fill)
            entry = artifact["models"].get(horizon)
            if entry is None:
                entry = artifact["models"].get(str(horizon))
            if entry is None:
                raise ValueError(f"horizon {horizon} has no model entry")
            if entry["kind"] == training.KIND_PERSISTENCE:
                value = last_level
            else:
                if entry["estimator"] is None:
                    raise ValueError(f"horizon {horizon} has no fitted estimator")
                delta = float(entry["estimator"].predict(row.to_frame().T)[0])
                value = last_level + delta
            target_date = last_date + pd.Timedelta(days=horizon)
            change = value - last_level
            direction = (
                "deeper" if change > 1e-9 else "shallower" if change < -1e-9 else "unchanged"
            )
            quantiles = artifact["metrics"]["by_horizon"][str(horizon)][
                "residual_quantiles"
            ]
            low = value + quantiles["p10"]
            high = value + quantiles["p90"]
            print(
                f"  h={horizon:>2} d  {target_date.date()}  {value:6.3f} m  "
                f"({change:+.3f} m, {direction} than the last observation) "
                f"[{entry['kind']}] band {low:6.3f}..{high:6.3f} m"
            )
    except Exception as error:  # noqa: BLE001 - reported as a failure below
        failures.append(f"inference path failed: {error}")

    print()
    print("Reported figures are held-out/backtest results for one dataset. They are not "
          "accuracy guarantees for future readings or other stations.")
    if failures:
        print()
        print("FAILURES:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("evaluation OK")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-evaluate the saved HydraSense model")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()
    raise SystemExit(evaluate(args.model, args.dataset))


if __name__ == "__main__":
    main()
