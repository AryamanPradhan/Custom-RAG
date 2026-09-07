"""SQLite schema and connection handling (Layer 09).

SQLite rather than Postgres because the working set is tiny - roughly six
fields per Property, a daily spend counter, one row per indexed Source, and
eval results. What it buys over config files is that onboarding a client is an
INSERT rather than a redeploy of the service every other client is being served
by.

Conversations are deliberately absent. Sessions are stateless: the widget
carries history, so there is no Visitor transcript here to retain or expire.
"""

from __future__ import annotations

from pathlib import Path

import aiosqlite

from app.logging_setup import get_logger

log = get_logger(__name__)

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS properties (
    property_id          TEXT PRIMARY KEY,
    display_name         TEXT NOT NULL,
    contact_route        TEXT,           -- JSON: {phone, email, url, note}
    daily_spend_cap_usd  REAL NOT NULL DEFAULT 5.0,
    last_ingested_at     TEXT,
    created_at           TEXT NOT NULL,
    active               INTEGER NOT NULL DEFAULT 1
);

-- One row per allowed origin. A dedicated table rather than a JSON column on
-- properties: the Origin header lookup happens on every single chat request,
-- and it must be an indexed point read, not a scan over every property.
CREATE TABLE IF NOT EXISTS property_origins (
    origin       TEXT PRIMARY KEY,
    property_id  TEXT NOT NULL REFERENCES properties(property_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS spend_ledger (
    property_id  TEXT NOT NULL REFERENCES properties(property_id) ON DELETE CASCADE,
    day          TEXT NOT NULL,          -- YYYY-MM-DD, UTC
    spent_usd    REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (property_id, day)
);

-- One row per indexed Source: which owner-supplied document is in the index,
-- at which content hash, since when. Answers "what do you hold about us?"
-- without reading the vector store back.
CREATE TABLE IF NOT EXISTS source_state (
    property_id      TEXT NOT NULL REFERENCES properties(property_id) ON DELETE CASCADE,
    uri              TEXT NOT NULL,
    content_hash     TEXT NOT NULL,
    indexed_at       TEXT NOT NULL,
    PRIMARY KEY (property_id, uri)
);

-- Account-wide usage counters, one row per UTC day. Persisted so that
-- restarting the process mid-session does not hand it a fresh budget.
CREATE TABLE IF NOT EXISTS usage_ledger (
    day        TEXT PRIMARY KEY,
    calls      INTEGER NOT NULL DEFAULT 0,
    spent_usd  REAL NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS feedback (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    property_id  TEXT NOT NULL,
    trace_id     TEXT,
    rating       TEXT NOT NULL,
    question     TEXT,
    comment      TEXT,
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS eval_runs (
    run_id           TEXT PRIMARY KEY,
    property_id      TEXT NOT NULL,
    label            TEXT,
    started_at       TEXT NOT NULL,
    cases            INTEGER NOT NULL DEFAULT 0,
    retrieval_hits   INTEGER NOT NULL DEFAULT 0,
    grounded         INTEGER NOT NULL DEFAULT 0,
    correct_deflect  INTEGER NOT NULL DEFAULT 0,
    notes            TEXT
);

CREATE TABLE IF NOT EXISTS eval_results (
    run_id        TEXT NOT NULL REFERENCES eval_runs(run_id) ON DELETE CASCADE,
    case_id       TEXT NOT NULL,
    question      TEXT NOT NULL,
    retrieved_hit INTEGER NOT NULL,
    grounded      INTEGER NOT NULL,
    deflected     INTEGER NOT NULL,
    answer        TEXT,
    detail        TEXT,
    PRIMARY KEY (run_id, case_id)
);

CREATE INDEX IF NOT EXISTS idx_feedback_property
    ON feedback(property_id, created_at);
"""


class Database:
    def __init__(self, path: str) -> None:
        self.path = path
        self._conn: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self.path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.executescript(SCHEMA)
        await self._conn.commit()
        log.info(f"SQLite ready at {self.path}.", path=self.path)

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Database.connect() has not been awaited")
        return self._conn


_db: Database | None = None


def get_db() -> Database:
    if _db is None:
        raise RuntimeError("database not initialised - call init_db() in the app lifespan")
    return _db


async def init_db(path: str) -> Database:
    global _db
    _db = Database(path)
    await _db.connect()
    return _db


async def close_db() -> None:
    global _db
    if _db is not None:
        await _db.close()
        _db = None
