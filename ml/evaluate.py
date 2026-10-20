"""Independently re-evaluate a saved HydraSense model artifact.

Run from the repository root (the server must be stopped only if you want to
compare, it is not required):

    backend/.venv/Scripts/python.exe ml/evaluate.py

The script reloads the artifact and the raw dataset and then recomputes

* the recency-weighted rolling-origin backtest for every candidate,
* the read-only hold-out score for every candidate,
* the per-horizon selection decision,
* one served forecast through the same code path the API uses,

and fails (exit status 1) if any recorded number does not reproduce, if the
selection does not follow the documented rule, or if a served model cannot be
loaded and applied. Nothing is accepted on trust (SRS 6.2, 9.7).
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


def evaluate(model_path: Path, dataset_path: Path) -> int:
    failures: list[str] = []
    artifact = joblib.load(model_path)
    frame, _ = dp.clean(dp.load_raw(dataset_path))

    end = pd.Timestamp(frame[dp.DATE_COLUMN].max()).normalize()
    do_fill = float(artifact["do_fill_value"])
    horizon_days = int(artifact["horizon_days"])
    holdout_start = end - pd.Timedelta(days=training.HOLDOUT_DAYS - 1)
    windows = training.backtest_windows(frame, holdout_start)
    weights = training.recency_weights(len(windows))

    print("HydraSense model evaluation")
    print("=" * 100)
    print(f"artifact      : {model_path}")
    print(f"trained at    : {artifact['trained_at']}")
    print(f"library       : trained with scikit-learn "
          f"{artifact['library_versions']['scikit-learn']}, reloaded with {sklearn.__version__}")
    print(f"horizon       : 1..{horizon_days} days")
    print(f"features      : {len(artifact['feature_order'])}")
    print(f"backtest      : {[[str(a), str(b)] for a, b in
                             ((w[0].date(), w[1].date()) for w in windows)]}")
    print(f"hold-out      : {holdout_start.date()} -> {end.date()} (read-only)")
    print(f"trained clean : {artifact['cleaning']['source_rows']} source rows, "
          f"{artifact['dataset']['rows']} usable")

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

    print()
    print(f"{'h':>3} {'served':>12} {'bt base':>9} {'bt forest':>10} {'ho base':>9} "
          f"{'ho forest':>10} {'sel ok':>7} {'match':>6}")
    print("-" * 100)

    reproduced = True
    for horizon in range(1, horizon_days + 1):
        data = dp.build_supervised(
            frame, horizon, do_fill_value=do_fill, target_mode=dp.TARGET_MODE_DELTA
        )
        target_dates = pd.to_datetime(data.target_dates)
        actual = dp.to_levels(data.origin_levels, data.y.to_numpy(), data.target_mode)
        persistence_all = data.origin_levels.to_numpy(dtype=float)

        fold_scores: dict[str, list[float]] = {kind: [] for kind in training.CANDIDATES}
        for window_start, window_end in windows:
            train_idx = np.where(target_dates < window_start)[0]
            eval_idx = np.where(
                (target_dates >= window_start) & (target_dates <= window_end)
            )[0]
            fold_scores[training.KIND_PERSISTENCE].append(
                dp.regression_metrics(actual[eval_idx], persistence_all[eval_idx])["mae"]
            )
            for kind in training.LEARNED_KINDS:
                estimator = training.build_estimator(kind)
                estimator.fit(data.X.iloc[train_idx], data.y.iloc[train_idx])
                predicted = dp.to_levels(
                    data.origin_levels, estimator.predict(data.X), data.target_mode
                )
                fold_scores[kind].append(
                    dp.regression_metrics(actual[eval_idx], predicted[eval_idx])["mae"]
                )
        backtest = {
            kind: float(np.average(fold_scores[kind], weights=weights))
            for kind in training.CANDIDATES
        }

        holdout_idx = np.where(target_dates >= holdout_start)[0]
        pre_holdout_idx = np.where(target_dates < holdout_start)[0]
        holdout = {
            training.KIND_PERSISTENCE: dp.regression_metrics(
                actual[holdout_idx], persistence_all[holdout_idx]
            )["mae"]
        }
        for kind in training.LEARNED_KINDS:
            estimator = training.build_estimator(kind)
            estimator.fit(data.X.iloc[pre_holdout_idx], data.y.iloc[pre_holdout_idx])
            predicted = dp.to_levels(
                data.origin_levels, estimator.predict(data.X), data.target_mode
            )
            holdout[kind] = dp.regression_metrics(actual[holdout_idx], predicted[holdout_idx])[
                "mae"
            ]

        # Reapply the documented selection rule.
        selected = min(backtest, key=lambda kind: backtest[kind])
        if selected != training.KIND_PERSISTENCE and backtest[selected] > backtest[
            training.KIND_PERSISTENCE
        ] * (1 - training.SELECTION_MARGIN):
            selected = training.KIND_PERSISTENCE

        stored = artifact["metrics"]["by_horizon"][str(horizon)]
        selection_ok = selected == stored["selected"]
        numbers_match = all(
            abs(backtest[kind] - stored["backtest"][kind]) < TOLERANCE
            for kind in training.CANDIDATES
        ) and all(
            abs(holdout[kind] - stored["holdout"][kind]) < TOLERANCE
            for kind in training.CANDIDATES
        )
        reproduced = reproduced and selection_ok and numbers_match
        if not selection_ok:
            failures.append(
                f"horizon {horizon}: selection rule gives {selected!r} but the artifact "
                f"stores {stored['selected']!r}"
            )
        if not numbers_match:
            failures.append(f"horizon {horizon}: recomputed metrics differ from stored values")

        print(
            f"{horizon:>3} {stored['selected']:>12} {backtest[training.KIND_PERSISTENCE]:>9.4f} "
            f"{backtest[training.KIND_FOREST]:>10.4f} "
            f"{holdout[training.KIND_PERSISTENCE]:>9.4f} "
            f"{holdout[training.KIND_FOREST]:>10.4f} "
            f"{'yes' if selection_ok else 'NO':>7} {'yes' if numbers_match else 'NO':>6}"
        )

    overall = artifact["metrics"]["overall"]
    print("-" * 100)
    print(f"all stored metrics and selections reproduced: {reproduced}")
    print(f"served backtest MAE mean {overall['selected_backtest_mae_mean']:.4f} m vs "
          f"persistence {overall['persistence_backtest_mae_mean']:.4f} m")
    print(f"served hold-out MAE mean {overall['selected_holdout_mae_mean']:.4f} m vs "
          f"persistence {overall['persistence_holdout_mae_mean']:.4f} m vs forest "
          f"{overall['forest_holdout_mae_mean']:.4f} m")
    print(f"learned candidates served for horizons: {overall['learned_horizons'] or 'none'}")

    # --- inference smoke test: exactly what the API does ------------------------
    print()
    print("inference smoke test (forecast issued from the last observed day)")
    last_date = pd.Timestamp(frame[dp.DATE_COLUMN].iloc[-1])
    last_level = float(frame[dp.TARGET_COLUMN].iloc[-1])
    print(f"last observation: {last_date.date()} -> {last_level:.3f} {artifact['unit']}")
    try:
        for horizon in (1, 7, int(horizon_days / 2), horizon_days):
            row = dp.inference_row(frame, horizon, do_fill_value=do_fill)
            # `models` is keyed by int horizon (train.py writes int keys); fall back to str.
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
            print(
                f"  h={horizon:>2} d  {target_date.date()}  {value:6.3f} m  "
                f"({change:+.3f} m, {direction} than the last observation) "
                f"[{entry['kind']}]"
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
