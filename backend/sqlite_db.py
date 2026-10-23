"""SQLite local storage for HydraSense preferences and optional cache.

Accessed only by the backend process running on the relevant device.
Never accessed from browser JavaScript.
"""

import sqlite3
import json
import os
from pathlib import Path
from typing import Any, Optional


# Path relative to the repository root
DB_PATH = Path(__file__).resolve().parent.parent / ".hydrasense.db"


def _get_conn():
    """Open a new SQLite connection and enable the JSON extension."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create the required tables if they do not already exist."""
    conn = _get_conn()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS preferences (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            )
        """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cached_results (
                key TEXT PRIMARY KEY,
                result_json TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                updated_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            )
        """
        )
        conn.commit()
    finally:
        conn.close()


def set_preference(key: str, value: Any) -> None:
    """Store a simple string preference. Non-string values are JSON-serialized."""
    conn = _get_conn()
    try:
        serialized = json.dumps(value) if not isinstance(value, str) else value
        conn.execute(
            "INSERT OR REPLACE INTO preferences (key, value) VALUES (?, ?)",
            (key, serialized),
        )
        conn.commit()
    finally:
        conn.close()


def get_preference(key: str, default: Optional[Any] = None) -> Any:
    """Retrieve a stored preference; returns *default* if the key is absent."""
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT value FROM preferences WHERE key = ?", (key,)
        ).fetchone()
        if row is None or not row["value"]:
            return default
        # Values are stored as plain strings, not always as JSON, so decoding
        # must fall back to the raw text instead of raising.
        try:
            return json.loads(row["value"])
        except (TypeError, ValueError):
            return row["value"]
    finally:
        conn.close()


def invalidate_preference(key: str) -> None:
    """Remove a preference key entirely."""
    conn = _get_conn()
    try:
        conn.execute("DELETE FROM preferences WHERE key = ?", (key,))
        conn.commit()
    finally:
        conn.close()


def set_cached_result(key: str, result: dict, ttl_seconds: int = 3600) -> None:
    """Store a cached result with a time-to-live (seconds)."""
    conn = _get_conn()
    from datetime import datetime, timezone, timedelta

    expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO cached_results (key, result_json, expires_at)
            VALUES (?, ?, ?)
            """,
            (key, json.dumps(result), expires_at.isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def get_cached_result(key: str, default: Optional[dict] = None) -> Optional[dict]:
    """Retrieve a cached result if it has not expired; otherwise returns *default*."""
    conn = _get_conn()
    from datetime import datetime, timezone

    try:
        row = conn.execute(
            "SELECT result_json, expires_at FROM cached_results WHERE key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return default
        expires_at_str = row["expires_at"]
        if expires_at_str:
            expires_at = datetime.fromisoformat(expires_at_str)
            if datetime.now(timezone.utc) > expires_at:
                # Expired — treat as not found
                return default
        result = json.loads(row["result_json"]) if row["result_json"] else None
        return result if result is not None else default
    finally:
        conn.close()
        