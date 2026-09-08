"""Operator surface: onboarding, ingestion, metrics.

Admin-key guarded and never reachable from the widget. Ingestion runs inline
rather than in a task queue - a document takes seconds to index, this is
operated by hand, and a queue would be infrastructure with no second user.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

from app.api.deps import get_repo, require_admin
from app.config import get_settings
from app.gateway.budget import get_usage_limiter
from app.logging_setup import get_logger
from app.models.schemas import IngestSummary, PropertyIn, PropertyOut
from app.observability.metrics import METRICS
from app.storage.chat_log import ChatLog
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
        last_ingested_at=prop.last_ingested_at,
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
        f"Registered {prop.property_id} ({prop.display_name}).",
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
    log.warning(f"Deleted the whole corpus for {property_id}.", property_id=property_id)


# -- chat log --------------------------------------------------------------


@router.get("/properties/{property_id}/chats")
async def list_chats(
    property_id: str,
    limit: int = 50,
    session_id: str | None = None,
    deflected: bool = False,
) -> list[dict]:
    """What the Guide has been asked, newest first.

    `deflected=true` is the one an operator should read weekly: every turn the
    Corpus could not answer, which is a list of the documents a client has not
    sent yet.
    """
    return await ChatLog(get_db()).recent(
        property_id, limit=limit, session_id=session_id, deflected_only=deflected
    )


@router.delete("/properties/{property_id}/chats", status_code=204)
async def delete_chats(property_id: str) -> None:
    """Erasure, for transcripts rather than the Corpus.

    Separate from deleting the Corpus on purpose: an owner replacing their
    documents is not asking for their visitors' questions to be thrown away,
    and someone asking for the questions to go is not asking to be un-indexed.
    """
    deleted = await ChatLog(get_db()).delete_property(property_id)
    log.warning(
        f"Deleted {deleted} chat log rows for {property_id}.",
        property_id=property_id,
        rows=deleted,
    )


# -- ingestion -------------------------------------------------------------


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
    await repo.mark_ingested(property_id)
    return IngestSummary(**report.__dict__)


# -- observability ---------------------------------------------------------


@router.get("/metrics")
async def metrics() -> dict:
    """Counters plus today's account usage against its caps."""
    return {**METRICS.snapshot(), "usage": get_usage_limiter().snapshot()}
