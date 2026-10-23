"""Model service for HydraSense: load, validate, and apply the trained model.

This module is the single source of truth for how the saved artifact is
loaded and used for inference. It mirrors the same preprocessing that
``ml/train.py`` and ``ml/evaluate.py`` use (SRS 3.1, 9.5, 9.7).

The artifact is a *per-horizon* bundle: ``artifact["models"][h]`` holds the
candidate served for ``h`` days ahead (``persistence`` or a fitted
``RandomForest``/``Ridge`` that predicts a change from the origin level).
A forecast for an arbitrary horizon is therefore built by evaluating every
served horizon from 1..N from the same forecast origin (the last observed
day). That is a direct multi-step forecast, not a recursive one.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import pandas as pd

from ml import data_preprocessing as dp

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "ml" / "model.pkl"
DATASET_PATH = ROOT / "data" / "DWLR_Dataset_2023.csv"

KIND_PERSISTENCE = "persistence"
KIND_RIDGE = "ridge_delta"

_ARTIFACT_CACHE: dict[str, object] = {"mtime": None, "artifact": None}
_LOAD_ERROR: str | None = None


class ModelError(ValueError):
    """Raised when the artifact cannot be used for inference."""


def _dig(mapping: dict, key: int):
    """Look a horizon up with int or str keys (the artifact uses int keys)."""
    if key in mapping:
        return mapping[key]
    return mapping.get(str(key))


def load_artifact(refresh: bool = False) -> dict | None:
    """Load the trained model artifact (cached by file mtime)."""
    global _LOAD_ERROR
    if not MODEL_PATH.exists():
        _LOAD_ERROR = f"Model artifact not found at {MODEL_PATH}"
        return None
    try:
        mtime = MODEL_PATH.stat().st_mtime
    except OSError as exc:  # pragma: no cover - filesystem edge case
        _LOAD_ERROR = f"Model artifact is not readable: {exc}"
        return None
    if not refresh and _ARTIFACT_CACHE["mtime"] == mtime:
        return _ARTIFACT_CACHE["artifact"]  # type: ignore[return-value]
    try:
        artifact = joblib.load(MODEL_PATH)
    except Exception as exc:  # corrupted / incompatible pickle
        _LOAD_ERROR = f"Model artifact could not be loaded: {exc}"
        return None
    if not isinstance(artifact, dict) or "models" not in artifact:
        _LOAD_ERROR = "Model artifact has an unexpected structure (no 'models' entry)"
        return None
    _LOAD_ERROR = None
    _ARTIFACT_CACHE["mtime"] = mtime
    _ARTIFACT_CACHE["artifact"] = artifact
    return artifact


def load_error() -> str | None:
    """Return the reason the last :func:`load_artifact` call failed, if any."""
    return _LOAD_ERROR


def validate_artifact(artifact: dict) -> list[str]:
    """Check that the artifact is internally consistent.

    Returns a list of failure strings; an empty list means valid.
    """
    failures: list[str] = []

    expected = list(dp.FEATURE_ORDER)
    actual = list(artifact.get("feature_order", []))
    if actual != expected:
        failures.append(
            f"artifact feature_order ({len(actual)} features) does not match "
            f"data_preprocessing.FEATURE_ORDER ({len(expected)} features)"
        )

    dataset = artifact.get("dataset") or {}
    stored_sha = dataset.get("sha256", "")
    if DATASET_PATH.exists() and stored_sha:
        computed = hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest()
        if stored_sha != computed:
            failures.append(
                "dataset file differs from the one used at training time, so the "
                "saved model and its recorded metrics no longer describe it"
            )

    if not artifact.get("models"):
        failures.append("artifact contains no models")

    return failures


def served_horizons(artifact: dict) -> list[int]:
    """Sorted list of horizons that can be forecast with this artifact."""
    horizons = []
    for key in (artifact.get("models") or {}):
        try:
            horizons.append(int(key))
        except (TypeError, ValueError):
            continue
    return sorted(h for h in horizons if h >= 1)


def max_horizon(artifact: dict) -> int:
    horizons = served_horizons(artifact)
    if horizons:
        return horizons[-1]
    try:
        return int(artifact.get("horizon_days") or 0)
    except (TypeError, ValueError):
        return 0


def _model_kind(artifact: dict, horizon: int) -> str:
    served = artifact.get("served") or {}
    kind = _dig(served, horizon)
    if isinstance(kind, str):
        return kind
    entry = _dig(artifact.get("models") or {}, horizon) or {}
    return entry.get("kind") or KIND_PERSISTENCE


def _predict_one(
    artifact: dict, df: pd.DataFrame, horizon: int, do_fill: float
) -> tuple[float, str]:
    """Return (predicted level, model kind) for one horizon from the last origin."""
    kind = _model_kind(artifact, horizon)
    last_level = float(df[dp.TARGET_COLUMN].iloc[-1])

    if kind == KIND_PERSISTENCE:
        return last_level, kind

    entry = _dig(artifact.get("models") or {}, horizon) or {}
    estimator = entry.get("estimator")
    if estimator is None:
        raise ModelError(f"horizon {horizon} has no fitted estimator")

    row = dp.inference_row(df, horizon, do_fill_value=do_fill)
    try:
        delta = float(estimator.predict(row.to_frame().T)[0])
    except Exception as exc:
        raise ModelError(f"model prediction failed for horizon {horizon}: {exc}") from exc
    return last_level + delta, kind


def predict(artifact: dict | None, df: pd.DataFrame, horizon: int) -> dict:
    """Run inference for one horizon.

    Kept as a thin wrapper over :func:`forecast` so single-horizon callers get
    exactly the same numbers as the UI.
    """
    series = forecast(artifact, df, horizon)
    point = series["predictions"][-1]
    return {
        "predicted_level": point["predicted_level_m"],
        "change_from_last": series["change_m"],
        "direction": series["direction"],
        "model_kind": point["model_kind"],
    }


def _interval(artifact: dict, horizon: int) -> tuple[float, float] | None:
    """Empirical (p10, p90) residual offsets recorded for this horizon.

    Returns ``None`` for artifacts without uncertainty, so an older artifact
    keeps serving point forecasts instead of failing.
    """
    metrics = artifact.get("metrics") or {}
    by_horizon = metrics.get("by_horizon") or {}
    entry = _dig(by_horizon, horizon) or {}
    quantiles = entry.get("residual_quantiles") or {}
    try:
        return float(quantiles["p10"]), float(quantiles["p90"])
    except (KeyError, TypeError, ValueError):
        return None


def _insight(change_m: float, horizon: int, kinds: set[str]) -> str:
    if abs(change_m) < 0.005:
        headline = "expected to stay essentially flat"
    elif change_m > 0:
        headline = f"expected to deepen by {abs(change_m):.2f} m"
    else:
        headline = f"expected to recover by {abs(change_m):.2f} m"

    if kinds == {KIND_PERSISTENCE}:
        method = (
            "No learned model beat the persistence baseline for these horizons, "
            "so the forecast carries the latest observed level forward."
        )
    elif KIND_PERSISTENCE in kinds:
        # A mix of learned and persistence horizons: say so instead of
        # implying every day comes from a learned model.
        method = (
            "Each day comes from the model selected for that horizon: persistence "
            "where the regularised linear model failed to improve on it by at least "
            "the required margin, otherwise the per-horizon Ridge model. Selection "
            "used the rolling backtest only, never the held-out window."
        )
    else:
        method = (
            "Values come from the served per-horizon Ridge models, each chosen over "
            "persistence on a rolling backtest and evaluated on a held-out window."
        )
    days = "day" if horizon == 1 else "days"
    return (
        f"Over the next {horizon} {days} the water level is {headline} "
        f"relative to the last observation. {method} This is a model estimate, "
        "not an official groundwater measurement."
    )


def forecast(artifact: dict | None, df: pd.DataFrame, horizon: int) -> dict:
    """Build a direct multi-step forecast from the latest observation.

    Parameters
    ----------
    artifact:
        Loaded model bundle from :func:`load_artifact`.
    df:
        Cleaned dataset (ascending by ``Date``).
    horizon:
        Number of days ahead to forecast. Clamped to the served horizons.

    Returns
    -------
    dict with ``last_observed``, ``predictions`` (one entry per day), the
    overall ``change_m``/``direction`` and a human-readable ``insight``.
    """
    if artifact is None:
        raise ModelError(load_error() or "No trained model artifact is available")
    if df is None or df.empty:
        raise ModelError("No observations are available to forecast from")

    failures = validate_artifact(artifact)
    if failures:
        raise ModelError("; ".join(failures))

    df = df.sort_values(dp.DATE_COLUMN).reset_index(drop=True)
    if len(df) < dp.MIN_HISTORY_DAYS:
        raise ModelError(
            f"At least {dp.MIN_HISTORY_DAYS} observations are required to build "
            f"forecast features; the dataset has {len(df)}."
        )

    horizon = int(horizon)
    available = max_horizon(artifact)
    if horizon < 1:
        raise ModelError("horizon_days must be at least 1")
    if available and horizon > available:
        horizon = available

    do_fill = float(artifact.get("do_fill_value", 0.0))
    last_date = pd.Timestamp(df[dp.DATE_COLUMN].iloc[-1])
    last_level = float(df[dp.TARGET_COLUMN].iloc[-1])

    points: list[dict] = []
    kinds: set[str] = set()
    for h in range(1, horizon + 1):
        level, kind = _predict_one(artifact, df, h, do_fill)
        kinds.add(kind)
        point = {
            "date": (last_date + pd.Timedelta(days=h)).strftime("%Y-%m-%d"),
            "predicted_level_m": round(level, 3),
            "horizon_days": h,
            "model_kind": kind,
            "is_predicted": True,
        }
        band = _interval(artifact, h)
        if band is not None:
            # Empirical band from the backtest residuals of the served candidate.
            point["lower_m"] = round(level + band[0], 3)
            point["upper_m"] = round(level + band[1], 3)
        points.append(point)

    change = round(points[-1]["predicted_level_m"] - last_level, 3) if points else 0.0
    direction = "deeper" if change > 1e-9 else "shallower" if change < -1e-9 else "unchanged"

    return {
        "last_observed": {
            "date": last_date.strftime("%Y-%m-%d"),
            "level_m": round(last_level, 3),
        },
        "predictions": points,
        "horizon_days": horizon,
        "horizon_available": available,
        "change_m": change,
        "direction": direction,
        "model_kind": (
            "persistence" if kinds == {KIND_PERSISTENCE} else "per_horizon_selection"
        ),
        "insight": _insight(change, horizon, kinds),
        "metrics": artifact.get("metrics") or {},
        "uncertainty": artifact.get("uncertainty") or {},
        "unit": artifact.get("unit", dp.TARGET_UNIT),
        "unit_confirmed": bool(artifact.get("unit_confirmed", dp.UNIT_CONFIRMED)),
        "unit_note": artifact.get("unit_note", ""),
    }
