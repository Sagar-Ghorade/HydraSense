"""FastAPI backend for HydraSense."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
ML_DIR = ROOT / "ml"
DATA_DIR = ROOT / "data"
MODEL_PATH = ML_DIR / "model.pkl"
DATASET_PATH = DATA_DIR / "DWLR_Dataset_2023.csv"

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="HydraSense API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic models for API responses
# ---------------------------------------------------------------------------


class StationInfo(BaseModel):
    id: str = Field(..., description="Station identifier")
    name: str = Field(..., description="Station name")
    state: str = Field(..., description="State")
    district: str = Field(..., description="District")
    latitude: float | None = Field(default=None, description="Latitude")
    longitude: float | None = Field(default=None, description="Longitude")
    last_measured: str | None = Field(default=None, description="Last measured date")


class HistoricalReading(BaseModel):
    date: str = Field(..., description="Measurement date")
    water_level_m: float | None = Field(default=None, description="Water level in metres")
    rainfall_mm: float | None = Field(default=None, description="Rainfall in mm")
    temperature_c: float | None = Field(default=None, description="Temperature in °C")


class PredictionRequest(BaseModel):
    station_id: str = Field(..., description="Selected station identifier")
    horizon_days: int = Field(..., description="Forecast horizon in days (1-30)")


class PredictionPoint(BaseModel):
    date: str = Field(..., description="Forecast date")
    predicted_level_m: float = Field(..., description="Predicted water level in metres BGL")
    is_predicted: bool = Field(default=True, description="Whether this is predicted or observed")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    dataset_rows: int = Field(..., description="Number of dataset rows")
    model_trained: bool = Field(..., description="Whether a model is available")
    model_horizon_days: int | None = Field(default=None, description="Model forecast horizon")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def load_model():
    """Load the trained model artifact, or return None if not available."""
    if not MODEL_PATH.exists():
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None


def load_dataset() -> pd.DataFrame:
    """Load the raw DWLR dataset."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH, parse_dates=["Date"])
    return df


def get_station_for_series() -> dict:
    """Return illustrative station mapping for the single daily series.

    The DWLR dataset has no state/district/station columns — we provide a
    single illustrative mapping so the UI can show location selection without
    fabricating real metadata.
    """
    return {
        "station_id": "Pune (Station 1234)",
        "state": "Maharashtra",
        "district": "Pune",
        "name": "Pune Monitoring Station",
        "latitude": 18.5204,
        "longitude": 73.8567,
    }


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/health", response_model=HealthResponse)
def health():
    """Check whether the backend is running and report basic state."""
    dataset_rows = 0
    model_trained = False
    model_horizon_days = None

    try:
        df = load_dataset()
        dataset_rows = len(df)
    except FileNotFoundError:
        pass

    artifact = load_model()
    if artifact is not None:
        model_trained = True
        model_horizon_days = artifact.get("horizon_days")

    return HealthResponse(
        status="ok" if dataset_rows > 0 else "no_data",
        dataset_rows=dataset_rows,
        model_trained=model_trained,
        model_horizon_days=model_horizon_days,
    )


@app.get("/api/stations", response_model=list[StationInfo])
def get_stations():
    """List supported stations and location metadata."""
    station = get_station_for_series()
    return [StationInfo(
        id=station["station_id"],
        name=station["name"],
        state=station["state"],
        district=station["district"],
        latitude=station["latitude"],
        longitude=station["longitude"],
        last_measured="2023-12-31",
    )]


@app.get("/api/groundwater", response_model=list[HistoricalReading])
def get_groundwater():
    """Retrieve historical readings for a station and optional date range."""
    df = load_dataset()

    # The dataset is a single daily series; filter/limit as needed
    df = df.sort_values("Date").tail(365)  # last year for API response

    readings = []
    for _, row in df.iterrows():
        readings.append(HistoricalReading(
            date=row["Date"].strftime("%Y-%m-%d"),
            water_level_m=round(float(row["Water_Level_m"]), 3) if pd.notna(row["Water_Level_m"]) else None,
            rainfall_mm=round(float(row["Rainfall_mm"]), 1) if pd.notna(row["Rainfall_mm"]) else None,
            temperature_c=round(float(row["Temperature_C"]), 1) if pd.notna(row["Temperature_C"]) else None,
        ))
    return readings[:30]  # return most recent 30 days


@app.post("/api/predictions")
def predictions(request: PredictionRequest):
    """Return forecast values for a selected station and supported horizon."""
    artifact = load_model()
    if artifact is None:
        raise HTTPException(status_code=503, detail="No trained model available")

    # Load dataset and prepare inference
    df = load_dataset()

    # Get the DO fill value from the artifact, or compute default
    do_fill = artifact.get("do_fill_value", 0.0)

    # Use the last horizon days from the model's served configuration
    horizon = min(request.horizon_days, int(artifact.get("horizon_days", 30)))

    # Build inference row using the same preprocessing as training
    from ml.data_preprocessing import add_features, inference_row, to_levels, regression_metrics

    try:
        row = inference_row(df, horizon, do_fill_value=do_fill)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Forecast preparation failed: {e}")

    # Get the served model for this horizon
    # `models` is keyed by int horizon (train.py writes int keys); fall back to str.
    model_info = artifact["models"].get(horizon)
    if model_info is None:
        model_info = artifact["models"].get(str(horizon))
    if model_info is None:
        raise HTTPException(status_code=404, detail=f"No model trained for horizon {horizon}")
    kind = model_info["kind"]

    last_date = pd.Timestamp(df["Date"].iloc[-1])
    last_level = float(df["Water_Level_m"].iloc[-1])

    if kind == "persistence":
        predicted_level = last_level
    else:
        estimator = model_info["estimator"]
        if estimator is None:
            raise HTTPException(status_code=500, detail="Estimator not available - model may not be fitted")
        try:
            delta = float(estimator.predict(row.to_frame().T)[0])
            predicted_level = last_level + delta
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    # Generate prediction points for the chart/table
    from ml.data_preprocessing import build_supervised

    pred_data = build_supervised(df, horizon, do_fill_value=do_fill, target_mode="delta")
    target_dates = pd.to_datetime(pred_data.target_dates)
    origin_levels = pred_data.origin_levels

    points = []
    for i in range(min(horizon, len(target_dates))):
        p_date = target_dates.iloc[i]
        p_level = float(pred_data.y[i]) if i < len(pred_data.y) else None
        # Convert delta back to level if needed
        if p_level is not None and artifact["target_mode"] == "delta":
            p_level = last_level + p_level

        points.append(PredictionPoint(
            date=p_date.strftime("%Y-%m-%d"),
            predicted_level_m=round(p_level, 3) if p_level is not None else None,
            is_predicted=True,
        ))

    # Also include the last observed point
    points.insert(0, PredictionPoint(
        date=last_date.strftime("%Y-%m-%d"),
        predicted_level_m=round(last_level, 3),
        is_predicted=False,
    ))

    return {
        "station_id": request.station_id,
        "horizon_days": horizon,
        "unit": artifact.get("unit", "m"),
        "unit_confirmed": artifact.get("unit_confirmed", False),
        "unit_note": artifact.get("unit_note", ""),
        "last_observed": {
            "date": last_date.strftime("%Y-%m-%d"),
            "level_m": round(last_level, 3),
        },
        "predictions": points,
        "model_kind": kind,
    }