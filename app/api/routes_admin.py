"""Operator surface: onboarding, ingestion, drift, metrics.

Admin-key guarded and never reachable from the widget. Ingestion runs inline
rather than in a task queue - a crawl of a property site takes a couple of
minutes, this is operated by hand, and a queue would be infrastructure with no
second user.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

from app.api.deps import get_repo, require_admin
from app.config import get_settings
from app.ingestion.drift import check_drift
from app.logging_setup import get_logger
from app.models.schemas import (
    CrawlRequest,
    DriftReportOut,
    IngestSummary,
    PropertyIn,
    PropertyOut,
)
from app.observability.metrics import METRICS
from app.storage.db import get_db
from app.storage.properties import ContactRoute

log = get_logger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_admin)])


async def _to_out(prop, indexed: int | None = None) -> PropertyOut:
    repo = get_repo()
    return PropertyOut(
        property_id=prop.property_id,
        display_name=prop.display_name,
        allowed_origins=prop.allowed_origins,
        daily_spend_cap_usd=prop.daily_spend_cap_usd,
        spent_today_usd=round(await repo.spent_today(prop.property_id), 4),
        last_crawled_at=prop.last_crawled_at,
        active=prop.active,
        indexed_chunks=indexed,
    )


# -- properties ------------------------------------------------------------


@router.post("/properties", response_model=PropertyOut)
async def create_property(payload: PropertyIn) -> PropertyOut:
    repo = get_repo()
    prop = await repo.create(
        payload.property_id,
        payload.display_name,
        allowed_origins=payload.allowed_origins,
        contact_route=ContactRoute(**payload.contact_route.model_dump()),
        daily_spend_cap_usd=payload.daily_spend_cap_usd,
    )
    log.info(
        "property.created",
        property_id=prop.property_id,
        origins=prop.allowed_origins,
    )
    return await _to_out(prop)


@router.get("/properties", response_model=list[PropertyOut])
async def list_properties() -> list[PropertyOut]:
    return [await _to_out(p) for p in await get_repo().list_all()]


@router.get("/properties/{property_id}", response_model=PropertyOut)
async def get_property(property_id: str, request: Request) -> PropertyOut:
    prop = await get_repo().get(property_id)
    if prop is None:
        raise HTTPException(404, "No such property.")
    indexed = await request.app.state.store.count(property_id)
    return await _to_out(prop, indexed)


@router.delete("/properties/{property_id}", status_code=204)
async def delete_property_corpus(property_id: str, request: Request) -> None:
    """Layer 09 - erasure. Drops every indexed Chunk for this Property."""
    await request.app.state.store.delete_property(property_id)
    db = get_db()
    await db.conn.execute("DELETE FROM source_state WHERE property_id = ?", (property_id,))
    await db.conn.commit()
    log.warning("property.corpus_deleted", property_id=property_id)


# -- ingestion -------------------------------------------------------------


@router.post("/properties/{property_id}/crawl", response_model=IngestSummary)
async def crawl(property_id: str, payload: CrawlRequest, request: Request) -> IngestSummary:
    repo = get_repo()
    if await repo.get(property_id) is None:
        raise HTTPException(404, "Register the property before ingesting.")

    report = await request.app.state.ingestion.ingest_site(
        property_id,
        payload.start_url,
        max_pages=payload.max_pages,
        max_depth=payload.max_depth,
        include_paths=payload.include_paths,
        exclude_paths=payload.exclude_paths,
    )
    await repo.mark_crawled(property_id)
    return IngestSummary(**report.__dict__)


@router.post("/properties/{property_id}/upload", response_model=IngestSummary)
async def upload(
    property_id: str, request: Request, file: UploadFile = File(...)
) -> IngestSummary:
    settings = get_settings()
    repo = get_repo()
    if await repo.get(property_id) is None:
        raise HTTPException(404, "Register the property before ingesting.")

    # Read incrementally and abort past the limit: a plain file.read() would
    # buffer the entire body first, so a huge POST OOMs the process before the
    # 413 is ever raised.
    limit = settings.max_upload_mb * 1024 * 1024
    chunks: list[bytes] = []
    total = 0
    while piece := await file.read(1024 * 1024):
        total += len(piece)
        if total > limit:
            raise HTTPException(
                413, f"File exceeds the {settings.max_upload_mb} MB limit."
            )
        chunks.append(piece)
    data = b"".join(chunks)

    report = await request.app.state.ingestion.ingest_upload(
        property_id, file.filename or "upload", data
    )
    if report.errors and report.chunks == 0:
        raise HTTPException(422, "; ".join(report.errors))
    await repo.mark_crawled(property_id)
    return IngestSummary(**report.__dict__)


# -- drift -----------------------------------------------------------------


@router.post("/properties/{property_id}/drift", response_model=DriftReportOut)
async def drift(property_id: str) -> DriftReportOut:
    """Re-fetch indexed pages and report which changed.

    Embeds nothing. This is the cheap signal that a manually-refreshed corpus
    has fallen behind the live site.
    """
    report = await check_drift(get_db(), get_settings(), property_id)
    return DriftReportOut(
        property_id=report.property_id,
        checked=report.checked,
        changed=report.changed,
        unreachable=report.unreachable,
        is_stale=report.is_stale,
        summary=report.summary(),
        checked_at=report.checked_at,
    )


# -- observability ---------------------------------------------------------


@router.get("/metrics")
async def metrics() -> dict:
    return METRICS.snapshot()
