"""Corpus drift detection.

Re-crawls are manual and billable, which means a Property's corpus will lag its
live website - and a Guide citing last quarter's cancellation policy is
grounded in a source that is no longer true. The grounding check cannot catch
this: the answer *is* supported by the retrieved chunk.

So this runs the cheap half of a crawl and none of the expensive half. It
re-fetches pages and compares hashes; it never embeds anything. The output is a
report - "nine pages changed" - that turns silent staleness into a prompt to
refresh, and into a billable conversation with the client.
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime

import httpx

from app.config import Settings
from app.ingestion.crawler import extract_main_content
from app.logging_setup import get_logger
from app.storage.db import Database

log = get_logger(__name__)


@dataclass(slots=True)
class DriftReport:
    property_id: str
    checked: int = 0
    changed: list[str] = field(default_factory=list)
    unreachable: list[str] = field(default_factory=list)
    checked_at: str = ""

    @property
    def is_stale(self) -> bool:
        return bool(self.changed)

    def summary(self) -> str:
        if not self.checked:
            return "No indexed sources to check."
        if not self.changed and not self.unreachable:
            return f"All {self.checked} pages match the indexed corpus."
        parts = [f"{len(self.changed)} of {self.checked} pages changed"]
        if self.unreachable:
            parts.append(f"{len(self.unreachable)} unreachable")
        return "; ".join(parts) + "."


async def record_source_state(
    db: Database,
    property_id: str,
    uri: str,
    content_hash: str,
    *,
    etag: str | None = None,
    last_modified: str | None = None,
) -> None:
    """Called by the ingestion pipeline for every indexed Source."""
    now = datetime.now(UTC).isoformat()
    await db.conn.execute(
        """INSERT INTO source_state
           (property_id, uri, content_hash, etag, last_modified, indexed_at,
            last_checked_at, drifted)
           VALUES (?, ?, ?, ?, ?, ?, ?, 0)
           ON CONFLICT(property_id, uri) DO UPDATE SET
               content_hash = excluded.content_hash,
               etag = excluded.etag,
               last_modified = excluded.last_modified,
               indexed_at = excluded.indexed_at,
               last_checked_at = excluded.last_checked_at,
               drifted = 0""",
        (property_id, uri, content_hash, etag, last_modified, now, now),
    )
    await db.conn.commit()


async def check_drift(
    db: Database,
    settings: Settings,
    property_id: str,
    *,
    concurrency: int = 4,
) -> DriftReport:
    """Re-fetch indexed pages and flag the ones whose content changed.

    ETag and Last-Modified are checked first; only when neither settles it do
    we download the page and re-extract. Uploaded documents are skipped - they
    have no URL to poll and only change when the owner sends a new file.
    """
    cursor = await db.conn.execute(
        """SELECT uri, content_hash, etag, last_modified FROM source_state
           WHERE property_id = ? AND uri LIKE 'http%'""",
        (property_id,),
    )
    rows = await cursor.fetchall()
    report = DriftReport(
        property_id=property_id, checked_at=datetime.now(UTC).isoformat()
    )
    if not rows:
        return report

    semaphore = asyncio.Semaphore(concurrency)
    headers = {"User-Agent": settings.crawl_user_agent}

    async with httpx.AsyncClient(
        headers=headers, follow_redirects=True, timeout=20.0
    ) as client:

        async def check(row) -> tuple[str, str]:
            """Returns (uri, 'same' | 'changed' | 'unreachable')."""
            uri = row["uri"]
            async with semaphore:
                try:
                    conditional = {}
                    if row["etag"]:
                        conditional["If-None-Match"] = row["etag"]
                    if row["last_modified"]:
                        conditional["If-Modified-Since"] = row["last_modified"]

                    response = await client.get(uri, headers=conditional)
                    if response.status_code == 304:
                        return uri, "same"
                    if response.status_code >= 400:
                        return uri, "unreachable"

                    # Same extractor as ingestion, or the hashes can never
                    # agree and every page reports as changed forever.
                    extracted = extract_main_content(response.text)
                    if not extracted:
                        return uri, "unreachable"

                    fresh = hashlib.sha256(extracted.encode()).hexdigest()
                    return uri, "same" if fresh == row["content_hash"] else "changed"
                except httpx.HTTPError:
                    return uri, "unreachable"

        outcomes = await asyncio.gather(*(check(r) for r in rows))

    now = datetime.now(UTC).isoformat()
    for uri, verdict in outcomes:
        report.checked += 1
        if verdict == "changed":
            report.changed.append(uri)
        elif verdict == "unreachable":
            report.unreachable.append(uri)
        await db.conn.execute(
            """UPDATE source_state SET last_checked_at = ?, drifted = ?
               WHERE property_id = ? AND uri = ?""",
            (now, 1 if verdict == "changed" else 0, property_id, uri),
        )
    await db.conn.commit()

    log.info(
        "drift.checked",
        property_id=property_id,
        checked=report.checked,
        changed=len(report.changed),
        unreachable=len(report.unreachable),
    )
    return report
