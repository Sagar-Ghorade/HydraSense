"""PostgreSQL shared storage for HydraSense.

PostgreSQL is the shared source of truth for station metadata and prediction
history *when it is enabled* (SRS 2.4, 4.3). It is entirely optional: the API
works from the CSV dataset + SQLite cache without it, so this module must
never break startup when no database is configured.

Enable it by setting ``POSTGRES_DSN`` (see ``.env.example``). Importing this
module does not connect anywhere; the connection is made on first use.

All functions are synchronous wrappers around asyncpg so they can be called
from FastAPI's sync endpoints (which already run in a worker thread). No
credentials are ever returned to callers.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS stations (
        station_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        state TEXT NOT NULL,
        district TEXT NOT NULL,
        latitude DOUBLE PRECISION,
        longitude DOUBLE PRECISION,
        source_dataset TEXT,
        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS prediction_history (
        id BIGSERIAL PRIMARY KEY,
        station_id TEXT NOT NULL REFERENCES stations (station_id) ON DELETE CASCADE,
        user_sub TEXT NOT NULL,
        horizon_days INTEGER NOT NULL,
        predicted_level_m DOUBLE PRECISION NOT NULL,
        issued_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        note TEXT
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS prediction_history_station_issued_idx
        ON prediction_history (station_id, issued_at DESC)
    """,
    """
    CREATE TABLE IF NOT EXISTS model_version (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
)

_schema_ready = False


def dsn() -> str | None:
    """Return the configured DSN, or None when PostgreSQL is disabled."""
    value = os.getenv("POSTGRES_DSN", "").strip()
    return value or None


def is_enabled() -> bool:
    """True when a PostgreSQL DSN is configured."""
    return dsn() is not None


async def _connect():
    try:
        import asyncpg  # imported lazily: optional dependency
    except ImportError as exc:  # pragma: no cover - depends on install
        raise RuntimeError(
            "POSTGRES_DSN is set but the 'asyncpg' package is not installed. "
            "Run: pip install -r backend/requirements.txt"
        ) from exc
    return await asyncpg.connect(dsn())


def _run(coro):
    """Run a coroutine from synchronous code, with or without a running loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # A loop is already running in this thread: use a private one.
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def _ensure_schema() -> None:
    conn = await _connect()
    try:
        for statement in SCHEMA:
            await conn.execute(statement)
    finally:
        await conn.close()


def ensure_schema(force: bool = False) -> None:
    """Create the shared tables once per process."""
    global _schema_ready
    if _schema_ready and not force:
        return
    _run(_ensure_schema())
    _schema_ready = True


async def _record_prediction(
    station_id: str,
    user_sub: str,
    horizon_days: int,
    predicted_level_m: float,
    note: str | None,
) -> dict:
    await _ensure_schema()
    conn = await _connect()
    try:
        row = await conn.fetchrow(
            """
            INSERT INTO prediction_history
                (station_id, user_sub, horizon_days, predicted_level_m, note)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING id, station_id, user_sub, horizon_days,
                      predicted_level_m, issued_at, note
            """,
            station_id,
            user_sub,
            int(horizon_days),
            float(predicted_level_m),
            note,
        )
        return dict(row)
    finally:
        await conn.close()


def record_prediction(
    station_id: str,
    user_sub: str,
    horizon_days: int,
    predicted_level_m: float,
    note: str | None = None,
) -> dict:
    """Append one prediction to the shared history and return the stored row."""
    return _run(
        _record_prediction(
            station_id, user_sub, horizon_days, predicted_level_m, note
        )
    )


async def _get_prediction_history(
    station_id: str | None,
    user_sub: str | None,
    horizon_days: int | None,
    limit: int,
) -> list[dict]:
    await _ensure_schema()
    conn = await _connect()
    try:
        filters: list[str] = []
        params: list[Any] = []
        for column, value in (
            ("station_id", station_id),
            ("user_sub", user_sub),
            ("horizon_days", horizon_days),
        ):
            if value is not None:
                params.append(value)
                filters.append(f"{column} = ${len(params)}")
        where = ("WHERE " + " AND ".join(filters)) if filters else ""
        params.append(int(limit))
        rows = await conn.fetch(
            f"""
            SELECT id, station_id, user_sub, horizon_days,
                   predicted_level_m, issued_at, note
            FROM prediction_history
            {where}
            ORDER BY issued_at DESC
            LIMIT ${len(params)}
            """,
            *params,
        )
        return [dict(row) for row in rows]
    finally:
        await conn.close()


def get_prediction_history(
    station_id: str | None = None,
    user_sub: str | None = None,
    horizon_days: int | None = None,
    limit: int = 50,
) -> list[dict]:
    """Read prediction history, newest first."""
    return _run(
        _get_prediction_history(station_id, user_sub, horizon_days, limit)
    )


async def _upsert_station(
    station_id: str,
    name: str,
    state: str,
    district: str,
    latitude: float | None,
    longitude: float | None,
    source_dataset: str | None,
    metadata: dict | None,
) -> dict:
    await _ensure_schema()
    conn = await _connect()
    try:
        row = await conn.fetchrow(
            """
            INSERT INTO stations
                (station_id, name, state, district, latitude, longitude,
                 source_dataset, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb)
            ON CONFLICT (station_id) DO UPDATE SET
                name = EXCLUDED.name,
                state = EXCLUDED.state,
                district = EXCLUDED.district,
                latitude = EXCLUDED.latitude,
                longitude = EXCLUDED.longitude,
                source_dataset = EXCLUDED.source_dataset,
                metadata = EXCLUDED.metadata,
                updated_at = NOW()
            RETURNING station_id, name, state, district, latitude, longitude,
                      source_dataset, metadata, updated_at
            """,
            station_id,
            name,
            state,
            district,
            latitude,
            longitude,
            source_dataset,
            json.dumps(metadata or {}),
        )
        return dict(row)
    finally:
        await conn.close()


def upsert_station(
    station_id: str,
    name: str,
    state: str,
    district: str,
    latitude: float | None = None,
    longitude: float | None = None,
    source_dataset: str | None = None,
    metadata: dict | None = None,
) -> dict:
    """Insert or update one station record (idempotent)."""
    return _run(
        _upsert_station(
            station_id,
            name,
            state,
            district,
            latitude,
            longitude,
            source_dataset,
            metadata,
        )
    )
