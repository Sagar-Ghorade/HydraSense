"""HydraSense data preparation and feature engineering.

This module is the single source of truth for how the DWLR dataset is read,
cleaned and turned into model features. ``ml/train.py``, ``ml/evaluate.py`` and
``backend/model_service.py`` all import it, so training and inference cannot
drift apart in column order or preprocessing (SRS 3.2, 6.2).

Design notes
------------
* The DWLR file contains a single daily series and has **no** state, district or
  station columns. Location selection is handled by the API as an explicitly
  illustrative mapping onto this one series (:mod:`backend.main`).
* Every feature is known at the *forecast origin* (the last observation date).
  Future rainfall/temperature are never used, because they are not observed yet.
  The only future information used is the calendar day-of-year of the target
  date, which is known in advance.
* The target is ``Water_Level_m``. The unit is taken from the dataset column
  name; the depth datum (whether the column is really below-ground-level) is not
  documented in the source file, so it is reported as unconfirmed rather than
  assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DATE_COLUMN = "Date"
TARGET_COLUMN = "Water_Level_m"
RAINFALL_COLUMN = "Rainfall_mm"
TEMPERATURE_COLUMN = "Temperature_C"
DO_COLUMN = "Dissolved_Oxygen_mg_L"

#: Unit is only as trustworthy as the source column name. See SRS 1.4 / 6.1.
TARGET_UNIT = "m"
UNIT_CONFIRMED = False

#: Frozen feature order. The API validates this against the saved artifact.
FEATURE_ORDER: tuple[str, ...] = (
    "wl_lag1",
    "wl_lag2",
    "wl_lag3",
    "wl_lag7",
    "wl_lag14",
    "wl_roll7_mean",
    "wl_roll14_mean",
    "wl_roll30_mean",
    "wl_roll7_std",
    "wl_diff1",
    "wl_diff7",
    "rain_lag1",
    "rain_roll7_sum",
    "rain_roll30_sum",
    "temp_lag1",
    "temp_roll7_mean",
    "do_lag1",
    "doy_sin",
    "doy_cos",
)

MIN_HISTORY_DAYS = 30  # rows required before an origin can produce features

#: Target modes understood by :func:`build_supervised`.
TARGET_MODE_DELTA = "delta"
TARGET_MODE_LEVEL = "level"

#: Horizons trained and evaluated by ``train.py`` (days ahead).
EVALUATED_HORIZONS = tuple(range(1, 31))


@dataclass
class CleaningReport:
    """Auditable record of what cleaning did, without touching the source file."""

    source_rows: int = 0
    rows_after_drop_duplicate_dates: int = 0
    duplicate_dates_removed: int = 0
    rows_dropped_non_numeric_target: int = 0
    missing_before: dict[str, int] = field(default_factory=dict)
    missing_after: dict[str, int] = field(default_factory=dict)
    negative_rainfall_rows: int = 0
    non_monotonic_dates: bool = False
    gap_days: int = 0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "source_rows": self.source_rows,
            "rows_after_drop_duplicate_dates": self.rows_after_drop_duplicate_dates,
            "duplicate_dates_removed": self.duplicate_dates_removed,
            "rows_dropped_non_numeric_target": self.rows_dropped_non_numeric_target,
            "missing_before": self.missing_before,
            "missing_after": self.missing_after,
            "negative_rainfall_rows": self.negative_rainfall_rows,
            "non_monotonic_dates": self.non_monotonic_dates,
            "gap_days": self.gap_days,
            "notes": self.notes,
        }


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the source file as-is. The original file is never modified."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path, parse_dates=[DATE_COLUMN])
    if DATE_COLUMN not in df.columns:
        raise ValueError(f"Dataset is missing the {DATE_COLUMN!r} column")
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Dataset is missing the {TARGET_COLUMN!r} target column")
    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Return a cleaned *working copy* plus a report of every change.

    Cleaning is deliberately conservative: unparseable target values are dropped,
    duplicated dates keep the first record, and missing auxiliary values are left
    missing (imputation happens later, fitted on training data only).
    """
    report = CleaningReport()
    report.source_rows = len(df)
    report.missing_before = {c: int(n) for c, n in df.isna().sum().items() if n}

    work = df.copy()
    work[DATE_COLUMN] = pd.to_datetime(work[DATE_COLUMN], errors="coerce")
    work = work.dropna(subset=[DATE_COLUMN])
    work = work.sort_values(DATE_COLUMN)
    report.non_monotonic_dates = bool(work[DATE_COLUMN].diff().dropna().lt(pd.Timedelta(0)).any())

    before = len(work)
    work = work.drop_duplicates(subset=[DATE_COLUMN], keep="first")
    report.duplicate_dates_removed = before - len(work)
    report.rows_after_drop_duplicate_dates = len(work)

    target_numeric = pd.to_numeric(work[TARGET_COLUMN], errors="coerce")
    dropped = int(target_numeric.isna().sum())
    if dropped:
        work = work.loc[target_numeric.notna()].copy()
        target_numeric = target_numeric.loc[work.index]
        report.notes.append(
            f"{dropped} row(s) had a missing or non-numeric {TARGET_COLUMN} and were "
            "excluded from modelling (the target is never imputed)."
        )
    work[TARGET_COLUMN] = target_numeric.astype(float)
    report.rows_dropped_non_numeric_target = dropped

    if RAINFALL_COLUMN in work.columns:
        rainfall = pd.to_numeric(work[RAINFALL_COLUMN], errors="coerce")
        report.negative_rainfall_rows = int((rainfall < 0).sum())
        if report.negative_rainfall_rows:
            report.notes.append(
                f"{report.negative_rainfall_rows} negative {RAINFALL_COLUMN} value(s) were "
                "kept but flagged: they are physically implausible."
            )

    full_range = pd.date_range(work[DATE_COLUMN].min(), work[DATE_COLUMN].max(), freq="D")
    report.gap_days = int(len(full_range) - len(work))
    if report.gap_days:
        report.notes.append(
            f"{report.gap_days} calendar day(s) have no record; features use positional "
            "lags, so short gaps are treated as consecutive observations."
        )

    report.missing_after = {c: int(n) for c, n in work.isna().sum().items() if n}
    work = work.reset_index(drop=True)
    return work, report


def add_features(df: pd.DataFrame, do_fill_value: float | None = None) -> pd.DataFrame:
    """Add origin-known features.

    ``do_fill_value`` must come from training data (a train-only median) so that
    imputing the dissolved-oxygen column cannot leak information across splits.
    """
    work = df.copy()
    wl = work[TARGET_COLUMN].astype(float)

    for lag in (1, 2, 3, 7, 14):
        work[f"wl_lag{lag}"] = wl.shift(lag - 1)

    for window in (7, 14, 30):
        roll = wl.rolling(window)
        work[f"wl_roll{window}_mean"] = roll.mean()
    work["wl_roll7_std"] = wl.rolling(7).std()
    work["wl_diff1"] = wl - wl.shift(1)
    work["wl_diff7"] = wl - wl.shift(7)

    if RAINFALL_COLUMN in work.columns:
        rain = pd.to_numeric(work[RAINFALL_COLUMN], errors="coerce")
        work["rain_lag1"] = rain
        work["rain_roll7_sum"] = rain.rolling(7).sum()
        work["rain_roll30_sum"] = rain.rolling(30).sum()

    if TEMPERATURE_COLUMN in work.columns:
        temp = pd.to_numeric(work[TEMPERATURE_COLUMN], errors="coerce")
        work["temp_lag1"] = temp
        work["temp_roll7_mean"] = temp.rolling(7).mean()

    if DO_COLUMN in work.columns:
        do = pd.to_numeric(work[DO_COLUMN], errors="coerce")
        if do_fill_value is not None:
            do = do.fillna(float(do_fill_value))
        work["do_lag1"] = do

    doy = work[DATE_COLUMN].dt.dayofyear.astype(float)
    work["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    work["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    return work


def train_do_fill_value(df: pd.DataFrame, train_origins: np.ndarray) -> float:
    """Median dissolved oxygen over the *training* origins only."""
    if DO_COLUMN not in df.columns:
        return 0.0
    values = pd.to_numeric(df[DO_COLUMN], errors="coerce")
    usable = values.iloc[train_origins].dropna()
    if usable.empty:
        return float(values.dropna().median()) if values.notna().any() else 0.0
    return float(usable.median())


@dataclass
class Supervised:
    """Feature matrix and targets for one forecast horizon.

    ``y`` holds either the absolute level or the change from the origin level,
    depending on ``target_mode``. ``origin_levels`` is always kept so callers can
    convert a predicted change back into a level.
    """

    X: pd.DataFrame
    y: pd.Series
    origins: pd.Series  # origin dates (last observed day used for the forecast)
    target_dates: pd.Series  # date each row predicts
    horizon: int
    origin_levels: pd.Series
    target_mode: str = TARGET_MODE_DELTA


def build_supervised(
    df: pd.DataFrame,
    horizon: int,
    do_fill_value: float,
    target_mode: str = TARGET_MODE_DELTA,
) -> Supervised:
    """Build direct multi-step training rows for ``horizon`` days ahead.

    ``target_mode``:
      * ``delta`` (default) - predict the *change* from the origin level. Water
        level here trends strongly within the year, so held-out targets fall
        outside the range of training targets; a tree model cannot extrapolate
        to them, whereas a change value stays inside the observed range.
      * ``level`` - predict the absolute level (kept for comparison).
    """
    if target_mode not in (TARGET_MODE_DELTA, TARGET_MODE_LEVEL):
        raise ValueError(f"Unknown target_mode: {target_mode!r}")
    if horizon < 1:
        raise ValueError("horizon must be at least 1 day")
    featured = add_features(df, do_fill_value=do_fill_value)
    target = df[TARGET_COLUMN].astype(float)

    rows = []
    for origin in range(len(df)):
        target_index = origin + horizon
        if target_index >= len(df):
            break
        rows.append((origin, target_index))

    if not rows:
        raise ValueError("Not enough rows to build a supervised set for this horizon")

    origins_idx = np.array([r[0] for r in rows])
    target_idx = np.array([r[1] for r in rows])

    X = featured.loc[origins_idx, list(FEATURE_ORDER)].copy()
    X.index = origins_idx
    X = X.astype(float)
    complete = X.notna().all(axis=1)
    X = X.loc[complete]
    keep = X.index.to_numpy()

    levels = pd.Series(target.loc[target_idx].to_numpy(), index=origins_idx).loc[keep]
    origin_levels = pd.Series(target.loc[keep].to_numpy(), index=keep)
    y = (levels - origin_levels) if target_mode == TARGET_MODE_DELTA else levels
    origin_dates = pd.Series(featured.loc[keep, DATE_COLUMN].to_numpy(), index=keep)
    target_dates = pd.Series(featured.loc[keep + horizon, DATE_COLUMN].to_numpy(), index=keep)

    return Supervised(
        X=X.reset_index(drop=True),
        y=y.reset_index(drop=True),
        origins=origin_dates.reset_index(drop=True),
        target_dates=target_dates.reset_index(drop=True),
        horizon=horizon,
        origin_levels=origin_levels.reset_index(drop=True),
        target_mode=target_mode,
    )


#: Fixed evaluation windows (days of target dates, not of origins).
DEFAULT_TEST_DAYS = 60
DEFAULT_VAL_DAYS = 60
MIN_TRAIN_ORIGINS = 60


def split_supervised(
    sup: "Supervised",
    dataset_end: pd.Timestamp,
    test_days: int = DEFAULT_TEST_DAYS,
    val_days: int = DEFAULT_VAL_DAYS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chronologically split in terms of the *target date* of each row.

    A direct multi-step model has fewer usable origins for larger horizons, so a
    split on origin positions would move the evaluation window every horizon and
    make horizons incomparable. Splitting on target dates keeps every horizon
    scored on the same calendar window: the last ``test_days`` of the series are
    the test targets, the ``val_days`` before them are the validation targets,
    and everything earlier is training data.
    """
    target_dates = pd.to_datetime(sup.target_dates)
    test_start = pd.Timestamp(dataset_end).normalize() - pd.Timedelta(days=test_days - 1)
    val_start = test_start - pd.Timedelta(days=val_days)

    test_idx = np.where(target_dates >= test_start)[0]
    val_idx = np.where((target_dates >= val_start) & (target_dates < test_start))[0]
    train_idx = np.where(target_dates < val_start)[0]

    if len(train_idx) < MIN_TRAIN_ORIGINS or len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError(
            f"Not enough rows for fixed-window splits at horizon {sup.horizon}: "
            f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}"
        )
    return train_idx, val_idx, test_idx


def inference_row(df: pd.DataFrame, horizon: int, do_fill_value: float) -> pd.Series:
    """Feature row for a forecast issued from the most recent observation.

    Reuses :func:`add_features` so the values are produced by exactly the same
    code path as training rows.
    """
    featured = add_features(df, do_fill_value=do_fill_value)
    last = len(df) - 1
    row = featured.loc[last, list(FEATURE_ORDER)].astype(float)

    # Day-of-year of the *target* date, which is known in advance.
    target_date = pd.Timestamp(df[DATE_COLUMN].iloc[last]) + pd.Timedelta(days=horizon)
    doy = float(target_date.dayofyear)
    row["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    row["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    missing = row[row.isna()]
    if not missing.empty:
        raise ValueError(
            "Cannot forecast: missing origin features " + ", ".join(missing.index) + "."
        )
    return row


def persistence_baseline(df: pd.DataFrame, horizon: int) -> Supervised:
    """Naive baseline: carry the level observed at the origin forward ``horizon`` days."""
    wl = df[TARGET_COLUMN].astype(float)
    origins = np.arange(0, len(df) - horizon)
    target_idx = origins + horizon
    origin_levels = wl.iloc[origins].reset_index(drop=True)
    levels = wl.iloc[target_idx].reset_index(drop=True)
    return Supervised(
        X=pd.DataFrame(index=pd.RangeIndex(len(origins))),
        y=levels,
        origins=df[DATE_COLUMN].iloc[origins].reset_index(drop=True),
        target_dates=df[DATE_COLUMN].iloc[target_idx].reset_index(drop=True),
        horizon=horizon,
        origin_levels=origin_levels,
        target_mode=TARGET_MODE_LEVEL,
    )


def to_levels(origin_levels: pd.Series, predicted: np.ndarray, target_mode: str) -> np.ndarray:
    """Convert model output into absolute levels."""
    predicted = np.asarray(predicted, dtype=float)
    if target_mode == TARGET_MODE_DELTA:
        return origin_levels.to_numpy(dtype=float) + predicted
    return predicted


def regression_metrics(actual, predicted) -> dict[str, float]:
    """MAE, RMSE and R2 for a set of predictions."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    errors = predicted - actual
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    variance = float(np.sum((actual - actual.mean()) ** 2))
    r2 = float(1 - np.sum(errors**2) / variance) if variance > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2, "n": int(len(actual))}
