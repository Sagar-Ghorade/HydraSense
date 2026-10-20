"""Model service for HydraSense: load, validate, and apply the trained model.

This module is the single source of truth for how the saved artifact is
loaded and used for inference. It mirrors the same logic in
``ml/evaluate.py`` so that backend and evaluation stay in sync (SRS 3.2,
6.2, 9.7).
"""

from __future__ import annotations

import joblib
from pathlib import Path

import numpy as np
import pandas as pd

from ml import data_preprocessing as dp

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "ml" / "model.pkl"

KIND_PERSISTENCE = "persistence"
KIND_FOREST = "random_forest_delta"
KIND_RIDGE = "ridge_delta"

LEARNED_KINDS = (KIND_FOREST, KIND_RIDGE)


def load_artifact() -> dict | None:
    """Load the trained model artifact, or return None if not available."""
    if not MODEL_PATH.exists():
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None


def validate_artifact(artifact: dict) -> list[str]:
    """Check that the artifact is internally consistent.

    Returns a list of failure strings; empty list means valid.
    """
    failures: list[str] = []

    # Feature order must match data_preprocessing.FEATURE_ORDER
    expected = list(dp.FEATURE_ORDER)
    actual = list(artifact.get("feature_order", []))
    if actual != expected:
        failures.append(
            f"artifact feature_order ({len(actual)} features) != "
            f"data_preprocessing.FEATURE_ORDER ({len(expected)} features)"
        )

    # Dataset hash must match the one recorded at training time
    import hashlib
    ds_path = Path(ROOT / "data" / "DWLR_Dataset_2023.csv")
    if ds_path.exists():
        stored_sha = artifact.get("dataset", {}).get("sha256", "")
        computed = hashlib.sha256(ds_path.read_bytes()).hexdigest()
        if stored_sha != computed:
            failures.append(
                "dataset hash differs from the one recorded at training time; "
                "metrics below describe a different dataset"
            )

    return failures


def predict(artifact: dict | None, df: pd.DataFrame, horizon: int) -> dict:
    """Run inference for one horizon using the served artifact.

    Returns a dict with:
      - "predicted_level": float (m BGL)
      - "change_from_last": float (m, positive = deeper / negative = shallower)
      - "direction": "deeper" | "shallower" | "unchanged"
      - "model_kind": "persistence" | "random_forest_delta" | "ridge_delta"
    """
    if artifact is None:
        raise ValueError("No trained model artifact provided")

    # Validate feature order / dataset hash
    failures = validate_artifact(artifact)
    if failures:
        raise ValueError("; ".join(failures))

    do_fill = float(artifact.get("do_fill_value", 0.0))
    last_date = pd.Timestamp(df["Date"].iloc[-1])
    last_level = float(df["Water_Level_m"].iloc[-1])
    model_type = artifact.get("served", {}).get(str(horizon), KIND_PERSISTENCE)

    # ---- inference row (same preprocessing as training) ----
    from ml.data_preprocessing import inference_row

    row = inference_row(df, horizon, do_fill_value=do_fill)

    if model_type == KIND_PERSISTENCE:
        predicted_level = last_level
        change = 0.0
        direction = "unchanged"
    else:
        # `models` is keyed by int horizon (train.py writes int keys); fall back to str.
        entry = artifact["models"].get(horizon)
        if entry is None:
            entry = artifact["models"].get(str(horizon))
        if entry is None:
            raise ValueError(f"horizon {horizon} has no model entry")
        estimator = entry.get("estimator")
        if estimator is None:
            raise ValueError(f"horizon {horizon} has no fitted estimator")
        try:
            delta = float(estimator.predict(row.to_frame().T)[0])
        except Exception as exc:
            raise ValueError(f"model prediction failed: {exc}")
        predicted_level = last_level + delta
        change = delta
        direction = (
            "deeper" if change > 1e-9 else "shallower" if change < -1e-9 else "unchanged"
        )

    return {
        "predicted_level": round(predicted_level, 3),
        "change_from_last": round(change, 3),
        "direction": direction,
        "model_kind": model_type,
    }