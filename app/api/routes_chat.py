"""Layer 01/02 - the Visitor-facing surface.

Two endpoints over the same pipeline: an SSE stream for the widget, and a JSON
endpoint for anything that cannot consume a stream (server-side rendering, the
eval harness, curl).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.api.deps import require_admin, resolve_property
from app.logging_setup import get_logger
from app.models.schemas import ChatRequest, ChatResponse, CitationOut, FeedbackRequest
from app.observability.metrics import METRICS
from app.observability.tracing import end_trace, start_trace
from app.pipeline.answer import AnswerPipeline
from app.storage.db import get_db
from app.storage.properties import Property

log = get_logger(__name__)
router = APIRouter(tags=["chat"])


def _pipeline(request: Request) -> AnswerPipeline:
    return request.app.state.pipeline


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    request: Request,
    prop: Property = Depends(resolve_property),
) -> ChatResponse:
    """Non-streaming answer."""
    trace = start_trace(prop.property_id, payload.session_id or "-")
    METRICS.incr("chat.request", mode="json", property_id=prop.property_id)
    try:
        result = await _pipeline(request).answer(
            prop,
            payload.message,
            [t.model_dump() for t in payload.history],
        )
        log.info("chat.answered", **trace.to_dict(), deflected=result.deflected)
        return ChatResponse(
            answer=result.answer,
            citations=[CitationOut(**asdict(c)) for c in result.citations],
            deflected=result.deflected,
            grounded=result.grounded,
            trace_id=trace.trace_id,
        )
    finally:
        end_trace()


@router.post("/chat/stream")
async def chat_stream(
    payload: ChatRequest,
    request: Request,
    prop: Property = Depends(resolve_property),
) -> StreamingResponse:
    """Server-sent events.

    Event types the widget handles: token, citations, deflect, retract,
    replace, done. `retract` is the one that matters - it means the answer
    streamed but failed the grounding check, and what was shown must be
    replaced by the Deflection it carries.
    """
    trace = start_trace(prop.property_id, payload.session_id or "-")
    METRICS.incr("chat.request", mode="stream", property_id=prop.property_id)
    pipeline = _pipeline(request)
    history = [t.model_dump() for t in payload.history]

    async def events():
        try:
            async for event in pipeline.stream(prop, payload.message, history):
                # Client disconnects are common on a website widget - a visitor
                # closing the tab mid-answer should stop the work, not log an
                # error.
                if await request.is_disconnected():
                    METRICS.incr("chat.client_disconnected")
                    break
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:  # noqa: BLE001
            METRICS.incr("chat.stream_error")
            log.error("chat.stream_error", error=f"{type(exc).__name__}: {exc}")
            yield f"data: {json.dumps({'type': 'error', 'message': 'Something went wrong.'})}\n\n"
        finally:
            log.info("chat.answered", **trace.to_dict())
            end_trace()

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # stop nginx buffering the stream
        },
    )


@router.post("/feedback", status_code=204)
async def feedback(
    payload: FeedbackRequest,
    prop: Property = Depends(resolve_property),
) -> None:
    """Layer 12 - the only human-in-the-loop surface this design keeps.

    There is no escalation queue: the Guide does not hand off to staff. What
    this captures is whether an answer was useful, which is what feeds the
    eval set.
    """
    from datetime import datetime

    db = get_db()
    await db.conn.execute(
        """INSERT INTO feedback
           (property_id, trace_id, rating, question, comment, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            prop.property_id,
            payload.trace_id,
            payload.rating,
            payload.question,
            payload.comment,
            datetime.now(UTC).isoformat(),
        ),
    )
    await db.conn.commit()
    METRICS.incr("feedback.received", rating=payload.rating)


@router.get("/feedback/{property_id}", dependencies=[Depends(require_admin)])
async def list_feedback(property_id: str, limit: int = 100) -> list[dict]:
    db = get_db()
    cursor = await db.conn.execute(
        """SELECT trace_id, rating, question, comment, created_at FROM feedback
           WHERE property_id = ? ORDER BY created_at DESC LIMIT ?""",
        (property_id, min(limit, 500)),
    )
    return [dict(row) for row in await cursor.fetchall()]
