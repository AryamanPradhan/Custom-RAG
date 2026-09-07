"""Layer 03 - the record of what is indexed.

The Corpus is built only from what a property owner hands over: uploaded
documents and pasted text. Nothing is fetched from the live web, so there is no
page to re-poll and no drift to detect - what remains worth keeping is the
ledger itself. One row per Source says which artifact is in the index at which
content hash, which is what makes an erasure request answerable without reading
the whole vector store back.
"""

from __future__ import annotations

from datetime import UTC, datetime

from app.storage.db import Database


async def record_source_state(
    db: Database, property_id: str, uri: str, content_hash: str
) -> None:
    """Called by the ingestion pipeline once a Source is actually indexed.

    Written after the upsert, never before: a row here is the claim "this
    document is in the index at this hash", and recording it up front would
    make a failed embedding call look like a successful ingestion.
    """
    now = datetime.now(UTC).isoformat()
    await db.conn.execute(
        """INSERT INTO source_state (property_id, uri, content_hash, indexed_at)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(property_id, uri) DO UPDATE SET
               content_hash = excluded.content_hash,
               indexed_at = excluded.indexed_at""",
        (property_id, uri, content_hash, now),
    )
    await db.conn.commit()
