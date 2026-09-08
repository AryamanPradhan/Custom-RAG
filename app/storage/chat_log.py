"""The chat log (Layer 09).

One row per completed turn: what was asked, what was answered, what it cited,
and - when it did not answer - why. Sessions stay stateless; nothing here is
ever read back into a prompt (see ADR 0003). It is an operator record, written
after the visitor already has their answer, and read by a person looking at
what the Guide is actually being asked.

What it is for, in the order it gets used:

  * Deflections. A run of them on the same question is a gap in the Corpus,
    and the only place that shows up is here.
  * Eval material. Real questions beat invented ones, and `guide logs` is
    where the next eval case comes from.
  * Answering an owner who asks what their Guide told a visitor last Tuesday.

Two things follow from holding visitor text. The question is stored as the
guardrails left it, so a card number a visitor pasted is `[redacted]` here as
well as in the prompt. And rows expire: `CHAT_LOG_RETENTION_DAYS` bounds how
long a transcript lives, swept at startup.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from app.logging_setup import get_logger
from app.models.domain import TurnRecord
from app.storage.db import Database

log = get_logger(__name__)


class ChatLog:
    def __init__(self, db: Database) -> None:
        self._db = db

    async def record(self, turn: TurnRecord) -> None:
        await self._db.conn.execute(
            """INSERT INTO chat_log
               (property_id, session_id, trace_id, created_at, question, answer,
                mode, intent, deflected, grounded, blocked, reason, citations,
                latency_ms)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                turn.property_id,
                turn.session_id or None,
                turn.trace_id or None,
                datetime.now(UTC).isoformat(),
                turn.question,
                turn.answer,
                turn.mode,
                turn.intent,
                int(turn.deflected),
                int(turn.grounded),
                int(turn.blocked),
                turn.reason or None,
                json.dumps(turn.citations),
                round(turn.latency_ms, 1),
            ),
        )
        await self._db.conn.commit()

    async def recent(
        self,
        property_id: str,
        *,
        limit: int = 50,
        session_id: str | None = None,
        deflected_only: bool = False,
    ) -> list[dict]:
        """Newest first. `deflected_only` is the view worth having: the turns
        the Corpus could not answer are the ones that change what gets
        ingested next."""
        sql = ["SELECT * FROM chat_log WHERE property_id = ?"]
        args: list[object] = [property_id]
        if session_id:
            sql.append("AND session_id = ?")
            args.append(session_id)
        if deflected_only:
            sql.append("AND deflected = 1")
        sql.append("ORDER BY created_at DESC, id DESC LIMIT ?")
        args.append(max(1, min(limit, 500)))

        cursor = await self._db.conn.execute(" ".join(sql), tuple(args))
        rows = []
        for row in await cursor.fetchall():
            turn = dict(row)
            turn["citations"] = json.loads(turn["citations"] or "[]")
            for flag in ("deflected", "grounded", "blocked"):
                turn[flag] = bool(turn[flag])
            rows.append(turn)
        return rows

    async def delete_property(self, property_id: str) -> int:
        """Erasure: drop every transcript held for one Property."""
        cursor = await self._db.conn.execute(
            "DELETE FROM chat_log WHERE property_id = ?", (property_id,)
        )
        await self._db.conn.commit()
        return cursor.rowcount

    async def purge(self, older_than_days: int) -> int:
        """Drop transcripts past their retention window. 0 keeps everything.

        Runs at startup rather than on a timer: a long-lived process would need
        a scheduler, and a service that is restarted on every deploy gets swept
        often enough for a 90-day window.
        """
        if older_than_days <= 0:
            return 0
        cutoff = (datetime.now(UTC) - timedelta(days=older_than_days)).isoformat()
        cursor = await self._db.conn.execute(
            "DELETE FROM chat_log WHERE created_at < ?", (cutoff,)
        )
        await self._db.conn.commit()
        if cursor.rowcount:
            log.info(
                f"Purged {cursor.rowcount} chat log rows older than "
                f"{older_than_days} days.",
                rows=cursor.rowcount,
                retention_days=older_than_days,
            )
        return cursor.rowcount
