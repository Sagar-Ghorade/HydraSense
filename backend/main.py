"""FastAPI backend for HydraSense.

Serves the five reference screens' data through HTTP only (SRS 2.4): the
browser never touches the dataset, SQLite or PostgreSQL directly.

Endpoints
---------
GET  /api/health         service + dataset + model status
GET  /api/locations      state -> district -> station filter tree
GET  /api/stations       supported stations
GET  /api/groundwater    historical readings for a station/date range
POST /api/predictions    model forecast for a horizon
GET  /api/predictions    same as POST (query string) for easy inspection
GET  /api/metrics        held-out evaluation metrics stored in the artifact
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Read local configuration (POSTGRES_DSN et al.) before any optional
# integration is queried. A missing .env file is not an error.
try:  # python-dotenv is a declared dependency but the API works without it
    from dotenv import load_dotenv  # noqa: E402
except ImportError:  # pragma: no cover - depends on install
    pass
else:  # pragma: no cover - trivial
    load_dotenv(ROOT / ".env")

from ml import data_preprocessing as dp  # noqa: E402  (needs ROOT on path)

from backend import model_service, sqlite_db  # noqa: E402

DATASET_PATH = ROOT / "data" / "DWLR_Dataset_2023.csv"

# ---------------------------------------------------------------------------
# Station registry
# ---------------------------------------------------------------------------
# The DWLR file contains a single daily series and no state/district/station
# columns. Rather than inventing stations, the API exposes exactly one station
# and labels it illustrative, so the UI can exercise the location filters
# without presenting fabricated metadata as real (SRS 1.5, 11).
STATIONS: list[dict] = [
    {
        "id": "dwlr-001",
        "name": "DWLR Station 001 (single series)",
        "state": "Maharashtra",
        "district": "Pune",
        "latitude": 18.5204,
        "longitude": 73.8567,
        "illustrative": True,
        "note": (
            "The source file has no station column. This is the one series in "
            "the dataset; the location labels are placeholders for the UI."
        ),
    },
]

STATION_IDS = {s["id"] for s in STATIONS}

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Create the local SQLite tables on startup (SRS 2.1)."""
    sqlite_db.init_db()
    yield


app = FastAPI(
    title="HydraSense API",
    version="1.0.0",
    description=(
        "Groundwater level monitoring and prediction API. Values are model "
        "estimates, not official groundwater measurements."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Dataset access (cached by file mtime)
# ---------------------------------------------------------------------------
_DATASET_CACHE: dict[str, object] = {"mtime": None, "df": None}


def load_dataset() -> pd.DataFrame:
    """Load the source dataset, preserving the original records."""
    if not DATASET_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Dataset not found at {DATASET_PATH}",
        )
    mtime = DATASET_PATH.stat().st_mtime
    if _DATASET_CACHE["mtime"] == mtime and _DATASET_CACHE["df"] is not None:
        return _DATASET_CACHE["df"]  # type: ignore[return-value]
    try:
        df = dp.load_raw(DATASET_PATH)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Dataset could not be read: {exc}")
    df = df.sort_values(dp.DATE_COLUMN).drop_duplicates(
        subset=[dp.DATE_COLUMN], keep="first"
    )
    df = df.reset_index(drop=True)
    _DATASET_CACHE["mtime"] = mtime
    _DATASET_CACHE["df"] = df
    return df


def require_station(station_id: str) -> dict:
    for station in STATIONS:
        if station["id"] == station_id:
            return station
    raise HTTPException(
        status_code=404,
        detail=(
            f"Unknown station_id {station_id!r}. Supported stations: "
            + ", ".join(sorted(STATION_IDS))
        ),
    )


def _number(value, digits: int):
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    # "degraded": the model loads but fails validation (e.g. dataset changed).
    status: Literal["ok", "no_data", "no_model", "degraded"]
    dataset_rows: int
    dataset_start: str | None = None
    dataset_end: str | None = None
    model_trained: bool
    model_horizon_days: int | None = None
    model_trained_at: str | None = None
    unit: str = dp.TARGET_UNIT
    unit_confirmed: bool = dp.UNIT_CONFIRMED
    detail: str | None = None


class StationInfo(BaseModel):
    id: str
    name: str
    state: str
    district: str
    latitude: float | None = None
    longitude: float | None = None
    last_measured: str | None = None
    observations: int | None = None
    illustrative: bool = False
    note: str | None = None


class HistoricalReading(BaseModel):
    date: str
    water_level_m: float | None = None
    rainfall_mm: float | None = None
    temperature_c: float | None = None


class HistoricalResponse(BaseModel):
    station_id: str
    unit: str
    unit_confirmed: bool
    unit_note: str
    aggregate: Literal["none", "monthly"]
    count: int
    readings: list[HistoricalReading]


class PredictionPoint(BaseModel):
    date: str
    predicted_level_m: float
    horizon_days: int
    model_kind: str
    is_predicted: bool = True
    #: Empirical 80% band (p10..p90) from the backtest residuals of the
    #: candidate served for this horizon; absent for older artifacts.
    lower_m: float | None = None
    upper_m: float | None = None


class LastObserved(BaseModel):
    date: str
    level_m: float


class PredictionResponse(BaseModel):
    station_id: str
    horizon_days: int
    horizon_available: int
    unit: str
    unit_confirmed: bool
    unit_note: str
    last_observed: LastObserved
    predictions: list[PredictionPoint]
    change_m: float
    direction: Literal["deeper", "shallower", "unchanged"]
    model_kind: str
    insight: str
    metrics: dict
    uncertainty: dict = Field(default_factory=dict)
    cached: bool = False


class LocationNode(BaseModel):
    state: str
    districts: list[dict]


class PredictionRequest(BaseModel):
    station_id: str = Field(
        default=STATIONS[0]["id"], description="Selected station identifier"
    )
    horizon_days: int = Field(
        default=7, ge=1, le=90, description="Forecast horizon in days (1-90)"
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report whether the service, dataset and model artifact are usable."""
    try:
        df = load_dataset()
    except HTTPException as exc:
        return HealthResponse(
            status="no_data",
            dataset_rows=0,
            model_trained=model_service.load_artifact() is not None,
            detail=str(exc.detail),
        )

    artifact = model_service.load_artifact()
    detail = None
    status: Literal["ok", "no_model", "degraded"] = "ok"
    if artifact is None:
        detail = model_service.load_error()
        status = "no_model"
    else:
        failures = model_service.validate_artifact(artifact)
        if failures:
            # The file loads, but it does not describe the current dataset / code,
            # so predictions will be refused. Do not report a healthy status.
            detail = "; ".join(failures)
            status = "degraded"

    return HealthResponse(
        status=status,
        dataset_rows=int(len(df)),
        dataset_start=pd.Timestamp(df[dp.DATE_COLUMN].iloc[0]).strftime("%Y-%m-%d"),
        dataset_end=pd.Timestamp(df[dp.DATE_COLUMN].iloc[-1]).strftime("%Y-%m-%d"),
        model_trained=artifact is not None,
        model_horizon_days=model_service.max_horizon(artifact) if artifact else None,
        model_trained_at=(artifact or {}).get("trained_at"),
        unit=(artifact or {}).get("unit", dp.TARGET_UNIT),
        unit_confirmed=bool((artifact or {}).get("unit_confirmed", dp.UNIT_CONFIRMED)),
        detail=detail,
    )


@app.get("/api/stations", response_model=list[StationInfo])
def get_stations() -> list[StationInfo]:
    """List the supported station(s) with coverage metadata."""
    df = load_dataset()
    last_date = pd.Timestamp(df[dp.DATE_COLUMN].iloc[-1]).strftime("%Y-%m-%d")
    return [
        StationInfo(
            **{k: station[k] for k in ("id", "name", "state", "district")},
            latitude=station["latitude"],
            longitude=station["longitude"],
            last_measured=last_date,
            observations=int(len(df)),
            illustrative=bool(station["illustrative"]),
            note=station["note"],
        )
        for station in STATIONS
    ]


@app.get("/api/locations", response_model=list[LocationNode])
def get_locations() -> list[LocationNode]:
    """Return the state -> district -> station tree used by the filters."""
    grouped: dict[str, dict[str, list[dict]]] = {}
    for station in STATIONS:
        grouped.setdefault(station["state"], {}).setdefault(
            station["district"], []
        ).append({"id": station["id"], "name": station["name"]})

    return [
        LocationNode(
            state=state,
            districts=[
                {"district": district, "stations": stations}
                for district, stations in districts.items()
            ],
        )
        for state, districts in grouped.items()
    ]


@app.get("/api/groundwater", response_model=HistoricalResponse)
def get_groundwater(
    station_id: str = Query(default=STATIONS[0]["id"], description="Station id"),
    start: date | None = Query(default=None, description="Inclusive start date"),
    end: date | None = Query(default=None, description="Inclusive end date"),
    limit: int = Query(default=365, ge=1, le=3650, description="Max records"),
    aggregate: Literal["none", "monthly"] = Query(
        default="none", description="Return daily records or monthly means"
    ),
) -> HistoricalResponse:
    """Return historical observations for a station, oldest first."""
    require_station(station_id)
    if start and end and start > end:
        raise HTTPException(status_code=422, detail="start must be on or before end")

    df = load_dataset()
    work = df

    if start is not None:
        work = work[work[dp.DATE_COLUMN] >= pd.Timestamp(start)]
    if end is not None:
        work = work[work[dp.DATE_COLUMN] <= pd.Timestamp(end)]

    if work.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                "No observations exist in the requested date range "
                f"({start or 'first record'} to {end or 'last record'})"
            ),
        )

    if aggregate == "monthly":
        work = work.copy()
        work["month"] = work[dp.DATE_COLUMN].dt.to_period("M").dt.to_timestamp()
        grouped = (
            work.groupby("month")
            .agg(
                water_level_m=(dp.TARGET_COLUMN, "mean"),
                rainfall_mm=(dp.RAINFALL_COLUMN, "sum"),
                temperature_c=(dp.TEMPERATURE_COLUMN, "mean"),
            )
            .reset_index()
        )
        # ``limit`` is part of the endpoint contract for both aggregations.
        grouped = grouped.tail(limit)
        readings = [
            HistoricalReading(
                date=pd.Timestamp(row["month"]).strftime("%Y-%m"),
                water_level_m=_number(row["water_level_m"], 3),
                rainfall_mm=_number(row["rainfall_mm"], 1),
                temperature_c=_number(row["temperature_c"], 1),
            )
            for _, row in grouped.iterrows()
        ]
    else:
        work = work.tail(limit)
        readings = [
            HistoricalReading(
                date=pd.Timestamp(row[dp.DATE_COLUMN]).strftime("%Y-%m-%d"),
                water_level_m=_number(row[dp.TARGET_COLUMN], 3),
                rainfall_mm=_number(row[dp.RAINFALL_COLUMN], 1),
                temperature_c=_number(row[dp.TEMPERATURE_COLUMN], 1),
            )
            for _, row in work.iterrows()
        ]

    artifact = model_service.load_artifact() or {}
    return HistoricalResponse(
        station_id=station_id,
        unit=artifact.get("unit", dp.TARGET_UNIT),
        unit_confirmed=bool(artifact.get("unit_confirmed", dp.UNIT_CONFIRMED)),
        unit_note=artifact.get("unit_note", ""),
        aggregate=aggregate,
        count=len(readings),
        readings=readings,
    )


def _build_prediction(station_id: str, horizon_days: int) -> PredictionResponse:
    require_station(station_id)
    df = load_dataset()

    artifact = model_service.load_artifact()
    if artifact is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "No usable trained model: "
                + (model_service.load_error() or "artifact missing")
            ),
        )

    # The cache key must change whenever the dataset *or* the model changes,
    # otherwise a retrained artifact would keep serving stale forecasts. The
    # serving code is part of the key for the same reason: how a forecast is
    # built (fields, wording, uncertainty) lives in these modules, so changing
    # it must not keep serving payloads written by the previous implementation
    # until the TTL happens to expire.
    try:
        artifact_mtime = model_service.MODEL_PATH.stat().st_mtime_ns
    except OSError:  # pragma: no cover - artifact presence checked above
        artifact_mtime = 0
    try:
        code_mtime = max(
            Path(model_service.__file__).stat().st_mtime_ns,
            Path(__file__).stat().st_mtime_ns,
        )
    except OSError:  # pragma: no cover - sources exist wherever the app runs
        code_mtime = 0
    cache_key = (
        f"forecast:{station_id}:{horizon_days}:"
        f"{_DATASET_CACHE['mtime']}:{artifact_mtime}:{code_mtime}"
    )
    cached = sqlite_db.get_cached_result(cache_key)
    if cached:
        cached["cached"] = True
        return PredictionResponse(**cached)

    try:
        result = model_service.forecast(artifact, df, horizon_days)
    except model_service.ModelError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    payload = {
        "station_id": station_id,
        **result,
        "cached": False,
    }

    try:
        sqlite_db.set_cached_result(cache_key, payload, ttl_seconds=3600)
    except Exception:
        # Caching is an optimisation; never fail a forecast because of it.
        pass

    _record_prediction(station_id, payload, artifact)
    return PredictionResponse(**payload)


def _record_prediction(station_id: str, payload: dict, artifact: dict) -> None:
    """Best-effort shared prediction history, only when PostgreSQL is enabled."""
    try:
        from backend import postgres_db

        if not postgres_db.is_enabled():
            return
        # prediction_history.station_id references stations(station_id), so the
        # station row must exist before the first history insert.
        station = next((s for s in STATIONS if s["id"] == station_id), None)
        if station is None:
            return
        postgres_db.upsert_station(
            station_id=station_id,
            name=station["name"],
            state=station["state"],
            district=station["district"],
            latitude=station["latitude"],
            longitude=station["longitude"],
            source_dataset=str(DATASET_PATH.relative_to(ROOT)),
            metadata={
                "illustrative": station["illustrative"],
                "note": station["note"],
            },
        )
        last = payload["predictions"][-1] if payload["predictions"] else None
        if last is None:
            return
        postgres_db.record_prediction(
            station_id=station_id,
            user_sub="anonymous",
            horizon_days=payload["horizon_days"],
            predicted_level_m=last["predicted_level_m"],
            note=f"model_kind={payload['model_kind']}",
        )
    except Exception:
        # PostgreSQL is optional; the API must still answer without it.
        pass


@app.post("/api/predictions", response_model=PredictionResponse)
def create_prediction(request: PredictionRequest) -> PredictionResponse:
    """Forecast groundwater levels for the requested horizon."""
    return _build_prediction(request.station_id, request.horizon_days)


@app.get("/api/predictions", response_model=PredictionResponse)
def read_prediction(
    station_id: str = Query(default=STATIONS[0]["id"]),
    horizon_days: int = Query(default=7, ge=1, le=90),
) -> PredictionResponse:
    """Same forecast as the POST form, convenient for browsers and tests."""
    return _build_prediction(station_id, horizon_days)


@app.get("/api/metrics")
def get_metrics() -> dict:
    """Held-out evaluation metrics recorded in the model artifact."""
    artifact = model_service.load_artifact()
    if artifact is None:
        raise HTTPException(
            status_code=503,
            detail=model_service.load_error() or "No trained model available",
        )
    failures = model_service.validate_artifact(artifact)
    return {
        "trained_at": artifact.get("trained_at"),
        "model_type": artifact.get("model_type"),
        "target": artifact.get("target"),
        "target_mode": artifact.get("target_mode"),
        "horizon_days": model_service.max_horizon(artifact),
        "metrics": artifact.get("metrics") or {},
        "selection": artifact.get("selection") or {},
        "dataset": artifact.get("dataset") or {},
        "cleaning": artifact.get("cleaning") or {},
        "validation_warnings": failures,
    }
