"""PostgreSQL shared storage for HydraSense DWLR records, station metadata,
and prediction history.

Used as the shared source of truth when enabled. All queries must filter
by user.sub (the authenticated Freebuff user) to isolate data per account.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import asyncpg


# Environment variables (set by the user / deployment)
PG_DSN = os.getenv("POSTGRES_DSN", "postgresql://hydrasense:hydrasense@localhost/hydrasense")


# Ensure the DB and required tables exist the first time this module is imported.
# This runs synchronously at import time in the worker process.
def _ensure_schema() -> None:
    """Create tables if they don't exist yet."""
    import asyncio

    async def _run():
        conn = await asyncpg.connect(PG_DSN)
        try:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS stations (
                    station_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    state TEXT NOT NULL,
                    district TEXT NOT NULL,
                    latitude DOUBLE PRECISION,
                    longitude DOUBLE PRECISION,
                    source_dataset TEXT,
                    metadata JSONB DEFAULT '{}'::jsonb
                )
                """
            )
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS prediction_history (
                    id BIGSERIAL PRIMARY KEY,
                    station_id TEXT REFERENCES stations (station_id),
                    user_sub TEXT NOT NULL,
                    horizon_days INTEGER NOT NULL,
                    predicted_level_m DOUBLE PRECISION NOT NULL,
                    issued_at TIMESTAMPTZ DEFAULT NOW(),
                    note TEXT
                )
                """
            )
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS model_version (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
                """
            )
        finally:
            await conn.close()

    asyncio.get_event_loop().run_until_complete(_ensure_schema())


ensure_schema_called = False


def _ensure_schema_once() -> None:
    """Call _ensure_schema once per process lifetime."""
    global ensure_schema_called
    if not ensure_schema_called:
        _ensure_schema()
        ensure_schema_called = True


# ---------------------------------------------------------------------------
# Public API (called from backend.main or other modules)
# ---------------------------------------------------------------------------

_ensure_schema_once()


def record_prediction(
    station_id: str,
    user_sub: str,
    horizon_days: int,
    predicted_level_m: float,
    note: str | None = None,
) -> dict:
    """Record a prediction in the shared history.

    Returns the inserted row as a dict.
    """
    import asyncio

    async def _run() -> dict:
        conn = await asyncpg.connect(PG_DSN)
        try:
            row = await conn.fetchrow(
                """
                INSERT INTO prediction_history (station_id, user_sub, horizon_days, predicted_level_m, note)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING id, station_id, user_sub, horizon_days, predicted_level_m, issued_at, note
                """,
                station_id,
                user_sub,
                horizon_days,
                predicted_level_m,
                note,
            )
            return dict(row)
        finally:
            await conn.close()

    return asyncio.get_event_loop().run_until_complete(_run())


def get_prediction_history(
    station_id: str | None = None,
    user_sub: str | None = None,
    horizon_days: int | None = None,
    limit: int = 50,
) -> List[dict]:
    """Retrieve prediction history, optionally filtered.

    Results are always ordered by issued_at DESC.
    """
    import asyncio

    async def _run() -> List[dict]:
        conn = await asyncpg.connect(PG_DSN)
        try:
            qs: List[str] = []
            params: List[Any] = []
            p = 1

            if station_id is not None:
                qs.append(f"station_id = ${p}")
                params.append(station_id)
                p += 1
            if user_sub is not None:
                qs.append(f"user_sub = ${p}")
                params.append(user_sub)
                p += 1
            if horizon_days is not None:
                qs.append(f"horizon_days = ${p}")
                params.append(horizon_days)
                p += 1

            where = "WHERE " + " AND ".join(qs) if qs else ""

            rows = await conn.fetch(
                f"""
                SELECT id, station_id, user_sub, horizon_days, predicted_level_m, issued_at, note
                FROM prediction_history
                {where}
                ORDER BY issued_at DESC
                LIMIT ${p}
                """,
                *params,
                limit,
            )
            return [dict(r) for r in rows]
        finally:
            await conn.close()

    return asyncio.get_event_loop().run_until_complete(_run())


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
    """Upsert a station record and return the resulting row.

    Idempotent — safe to call on every startup.
    """
    import asyncio

    async def _run() -> dict:
        conn = await asyncpg.connect(PG_DSN)
        try:
            # Prepare metadata JSONB
            meta_json = json.dumps(metadata) if metadata else "{}"

            row = await conn.fetchrow(
                """
                INSERT INTO stations (station_id, name, state, district, latitude, longitude, source_dataset, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb)
                ON CONFLICT (station_id)
                DO UPDATE SET
                    name = EXCLUDED.name,
                    state = EXCLUDED.state,
                    district = EXCLUDED.district,
                    latitude = EXCLUDED.latitude,
                    longitude = EXCLUDED.longitude,
                    source_dataset = EXCLUDED.source_dataset,
                    metadata = EXCLUDED.metadata,
                    updated_at = NOW()
                RETURNING station_id, name, state, district, latitude, longitude, source_dataset, metadata
                """,
                station_id,
                name,
                state,
                district,
                latitude,
                longitude,
                source_dataset,
                meta_json,
            )
            return dict(row)
        finally:
            await conn.close()

    return asyncio.get_event_loop().run_until_complete(_run())