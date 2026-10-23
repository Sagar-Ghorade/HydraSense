"""Regression tests for the HydraSense ML pipeline.

Run from the repository root (also collectable by pytest):

    backend/.venv/Scripts/python.exe ml/test_pipeline.py

These pin down the invariants the pipeline depends on, in particular train/serve
parity of the feature row and the "no information from the future" property of
every feature. Those failures are silent: the pipeline runs, the metrics look
plausible, and the served model is simply fed inputs it never saw while fitting.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_preprocessing as dp  # noqa: E402
import train as training  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "DWLR_Dataset_2023.csv"

DO_FILL = 0.0


def _frame() -> pd.DataFrame:
    return dp.clean(dp.load_raw(DATASET))[0]


def _tiny_frame(days: int = 120) -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=days, freq="D")
    return pd.DataFrame(
        {
            "Date": dates,
            "Water_Level_m": 2.0 + np.linspace(0, 1, days),
            "Temperature_C": np.linspace(5, 15, days),
            "Rainfall_mm": np.linspace(0, 20, days),
            "pH": np.full(days, 7.0),
            "Dissolved_Oxygen_mg_L": np.full(days, 8.0),
        }
    )


# --- feature invariants -------------------------------------------------------


def test_feature_order_is_exactly_what_is_generated():
    frame = _tiny_frame()
    featured = dp.add_features(frame, do_fill_value=DO_FILL)
    missing = [c for c in dp.ORIGIN_FEATURES if c not in featured.columns]
    assert not missing, f"add_features does not create {missing}"
    supervised = dp.build_supervised(frame, 7, do_fill_value=DO_FILL)
    assert list(supervised.X.columns) == list(dp.FEATURE_ORDER)
    assert supervised.X.notna().all().all()
    # Seasonality is not an origin feature: it is attached per horizon.
    assert not set(dp.SEASONAL_FEATURES) & set(dp.ORIGIN_FEATURES)


def test_seasonality_uses_the_target_date_not_the_origin():
    frame = _tiny_frame()
    horizon = 7
    supervised = dp.build_supervised(frame, horizon, do_fill_value=DO_FILL)
    target_season = dp.seasonality(pd.to_datetime(supervised.target_dates).to_numpy())
    origin_season = dp.seasonality(pd.to_datetime(supervised.origins).to_numpy())
    for column in dp.SEASONAL_FEATURES:
        np.testing.assert_allclose(
            supervised.X[column].to_numpy(),
            target_season[column].to_numpy(),
            err_msg=f"{column} must encode the target date",
        )
    # Origins and targets differ, so an origin-encoded column would be caught.
    assert not np.allclose(
        supervised.X["doy_sin"].to_numpy(), origin_season["doy_sin"].to_numpy()
    )


def test_inference_row_matches_the_training_row_for_the_same_origin():
    """The served feature row must equal the training row for that origin."""
    frame = _frame()
    dates = pd.to_datetime(frame[dp.DATE_COLUMN])
    for horizon in (1, 7, 15, 30):
        supervised = dp.build_supervised(frame, horizon, do_fill_value=DO_FILL)
        origin_dates = pd.to_datetime(supervised.origins)
        for origin in (40, 120, 300, len(frame) - horizon - 1):
            origin_date = dates.iloc[origin]
            matches = np.where(origin_dates == origin_date)[0]
            if not len(matches):
                continue  # row had incomplete features and was dropped
            served = dp.inference_row(
                frame.iloc[: origin + 1], horizon, do_fill_value=DO_FILL
            )
            training_row = supervised.X.iloc[int(matches[0])]
            np.testing.assert_allclose(
                served.to_numpy(dtype=float),
                training_row.to_numpy(dtype=float),
                err_msg=(
                    f"train/serve feature mismatch at horizon {horizon}, "
                    f"origin {origin_date.date()}"
                ),
            )
            assert list(served.index) == list(dp.FEATURE_ORDER)


def test_no_feature_looks_into_the_future():
    """Recomputing on a truncated prefix must not change any origin feature."""
    frame = _frame()
    for origin in (45, 120, 300):
        full = dp.add_features(frame, do_fill_value=DO_FILL)
        prefix = dp.add_features(frame.iloc[: origin + 1], do_fill_value=DO_FILL)
        for column in dp.ORIGIN_FEATURES:
            left = float(full[column].iloc[origin])
            right = float(prefix[column].iloc[origin])
            assert abs(left - right) < 1e-12, (
                f"{column} at row {origin} depends on data after the origin "
                f"({left} vs {right})"
            )


def test_target_is_the_change_from_the_origin_level():
    frame = _frame()
    horizon = 5
    supervised = dp.build_supervised(
        frame, horizon, do_fill_value=DO_FILL, target_mode=dp.TARGET_MODE_DELTA
    )
    levels = frame.set_index(pd.to_datetime(frame[dp.DATE_COLUMN]))[dp.TARGET_COLUMN]
    expected = (
        levels.reindex(pd.to_datetime(supervised.target_dates)).to_numpy(dtype=float)
        - levels.reindex(pd.to_datetime(supervised.origins)).to_numpy(dtype=float)
    )
    np.testing.assert_allclose(supervised.y.to_numpy(dtype=float), expected)
    # Origin / target dates are exactly `horizon` days apart.
    gap = pd.to_datetime(supervised.target_dates) - pd.to_datetime(supervised.origins)
    assert (gap == pd.Timedelta(days=horizon)).all()


# --- cleaning -----------------------------------------------------------------


def test_cleaning_reports_unsorted_dates_before_sorting():
    shuffled = _tiny_frame().sample(frac=1.0, random_state=0).reset_index(drop=True)
    _, report = dp.clean(shuffled)
    assert report.non_monotonic_dates is True

    ordered = _tiny_frame()
    _, ordered_report = dp.clean(ordered)
    assert ordered_report.non_monotonic_dates is False


def test_cleaning_removes_duplicates_and_keeps_the_first():
    frame = _tiny_frame()
    duplicate = frame.iloc[[10]].copy()
    duplicate[dp.TARGET_COLUMN] = 99.0  # a marker value only the duplicate carries
    cleaned, report = dp.clean(pd.concat([frame, duplicate], ignore_index=True))
    assert report.duplicate_dates_removed == 1
    assert len(cleaned) == 120
    assert cleaned["Date"].is_monotonic_increasing
    assert cleaned["Date"].is_unique
    assert float(cleaned[dp.TARGET_COLUMN].iloc[10]) != 99.0


def test_cleaning_drops_non_numeric_targets_without_imputing():
    frame = _tiny_frame().astype({dp.TARGET_COLUMN: object})
    frame.loc[5, dp.TARGET_COLUMN] = "not-a-number"
    cleaned, report = dp.clean(frame)
    assert report.rows_dropped_non_numeric_target == 1
    assert len(cleaned) == 119
    assert cleaned[dp.TARGET_COLUMN].notna().all()


# --- evaluation protocol ------------------------------------------------------


def test_backtest_windows_end_the_day_before_the_hold_out():
    frame = _frame()
    end = pd.Timestamp(frame[dp.DATE_COLUMN].max()).normalize()
    holdout_start = end - pd.Timedelta(days=training.HOLDOUT_DAYS - 1)
    windows = training.backtest_windows(frame, holdout_start)

    assert len(windows) == training.BACKTEST_FOLDS
    assert windows == sorted(windows)
    assert windows[-1][1] == holdout_start - pd.Timedelta(days=1)
    for start, stop in windows:
        assert start < stop
        assert stop < holdout_start
        assert stop - start == pd.Timedelta(days=training.BACKTEST_FOLD_DAYS - 1)
    for (_, first), (second, _) in zip(windows, windows[1:]):
        assert second > first, "overlapping windows double-count targets"


def test_no_evaluation_row_was_seen_while_training():
    frame = _frame()
    end = pd.Timestamp(frame[dp.DATE_COLUMN].max()).normalize()
    holdout_start = end - pd.Timedelta(days=training.HOLDOUT_DAYS - 1)
    windows = training.backtest_windows(frame, holdout_start)
    supervised = dp.build_supervised(frame, 30, do_fill_value=DO_FILL)
    folds = training.fold_indices(supervised.target_dates, windows)
    assert len(folds) == training.BACKTEST_FOLDS
    target_dates = pd.to_datetime(supervised.target_dates)
    for (train_idx, eval_idx), (window_start, _) in zip(folds, windows):
        assert len(train_idx) and len(eval_idx)
        assert target_dates.iloc[train_idx].max() < pd.Timestamp(window_start)
        assert target_dates.iloc[eval_idx].min() >= pd.Timestamp(window_start)


def test_selection_rule_requires_the_gain_threshold():
    floor = 100.0
    # A marginal improvement must not displace the baseline.
    assert (
        training.select_kind(floor, floor * (1 - training.MIN_GAIN * 0.5))
        == training.KIND_PERSISTENCE
    )
    # Exactly at the threshold the learned candidate wins...
    assert (
        training.select_kind(floor, floor * (1 - training.MIN_GAIN))
        == training.KIND_RIDGE
    )
    # ...and anything worse stays on the baseline.
    assert training.select_kind(floor, floor * 1.01) == training.KIND_PERSISTENCE


def test_recency_weights_favour_the_most_recent_window():
    weights = training.recency_weights(6)
    assert np.isclose(weights.sum(), 1.0)
    assert np.all(np.diff(weights) > 0)


def test_regression_metrics_on_known_values():
    actual = np.array([1.0, 2.0, 3.0])
    predicted = np.array([1.0, 3.0, 4.0])
    metrics = dp.regression_metrics(actual, predicted)
    # errors are 0, +1, +1
    assert np.isclose(metrics["mae"], 2.0 / 3.0)
    assert np.isclose(metrics["rmse"], np.sqrt(2.0 / 3.0))
    assert metrics["n"] == 3


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"  ok    {test.__name__}")
        except Exception as error:  # noqa: BLE001 - reported below
            failures += 1
            print(f"  FAIL  {test.__name__}: {error}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
