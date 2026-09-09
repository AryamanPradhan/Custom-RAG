"""Layer 01/02 - the Visitor-facing surface.

Two endpoints over the same pipeline: an SSE stream for the widget, and a JSON
endpoint for anything that cannot consume a stream (server-side rendering, the
eval harness, curl).
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import StreamingResponse

from app.api.deps import admin_key_ok, require_admin, resolve_property
from app.api.sessions import resolve_session
from app.logging_setup import get_logger
from app.models.schemas import ChatRequest, ChatResponse, CitationOut, FeedbackRequest
from app.observability.metrics import METRICS
from app.observability.tracing import end_trace, start_trace, step_sink
from app.pipeline.answer import AnswerPipeline
from app.storage.db import get_db
from app.storage.properties import Property

log = get_logger(__name__)
router = APIRouter(tags=["chat"])


def _sse(event: dict) -> str:
    """SSE framing, written once."""
    return f"data: {json.dumps(event)}\n\n"


def _pipeline(request: Request) -> AnswerPipeline:
    return request.app.state.pipeline


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    request: Request,
    prop: Property = Depends(resolve_property),
) -> ChatResponse:
    """Non-streaming answer."""
    session = resolve_session(prop.property_id, payload.session_id)
    trace = start_trace(prop.property_id, session.thread_id)
    METRICS.incr("chat.request", mode="json", property_id=prop.property_id)
    try:
        result = await _pipeline(request).answer(
            prop,
            payload.message,
            [t.model_dump() for t in payload.history],
        )
        log.info(
            "Deflected." if result.deflected else "Answered.",
            **trace.to_dict(),
            deflected=result.deflected,
        )
        return ChatResponse(
            answer=result.answer,
            citations=[CitationOut(**asdict(c)) for c in result.citations],
            deflected=result.deflected,
            grounded=result.grounded,
            trace_id=trace.trace_id,
            session_id=session.token,
        )
    finally:
        end_trace()


@router.post("/chat/stream")
async def chat_stream(
    payload: ChatRequest,
    request: Request,
    prop: Property = Depends(resolve_property),
    x_admin_key: str | None = Header(default=None),
) -> StreamingResponse:
    """Server-sent events.

    Event types the widget handles: token, citations, deflect, retract,
    replace, done. `retract` is the one that matters - it means the answer
    streamed but failed the grounding check, and what was shown must be
    replaced by the Deflection it carries.

    A caller holding the admin key also receives `step` events as each stage
    of the pipeline finishes. That is for the operator console, and it is
    gated because step names, timings and candidate counts describe the shape
    of the machine rather than the answer - a Visitor has no use for them and
    should not be handed them.
    """
    session = resolve_session(prop.property_id, payload.session_id)
    trace = start_trace(prop.property_id, session.thread_id)
    METRICS.incr("chat.request", mode="stream", property_id=prop.property_id)
    pipeline = _pipeline(request)
    history = [t.model_dump() for t in payload.history]
    observed = admin_key_ok(x_admin_key)

    async def events():
        # First, before any model work: the token the Widget must echo on the
        # next turn. A stream that fails halfway still leaves the visitor in a
        # thread, which is the case where losing it would be most confusing.
        yield _sse({"type": "session", "session_id": session.token})

        # Steps complete while the pipeline is awaiting a model, not between
        # its yields, so draining them around `async for` would hold every
        # pre-generation stage back until the first token - the exact seconds
        # the console exists to fill. Both go into one queue instead, and the
        # pipeline runs as a task so a step reaches the client when it happens.
        queue: asyncio.Queue = asyncio.Queue()
        finished = object()

        async def pump() -> None:
            try:
                async for event in pipeline.stream(prop, payload.message, history):
                    await queue.put(event)
            except Exception as exc:  # noqa: BLE001
                METRICS.incr("chat.stream_error")
                log.error(
                    "The answer stream failed; the widget was told to retract.",
                    error=f"{type(exc).__name__}: {exc}",
                )
                await queue.put({"type": "error", "message": "Something went wrong."})
            finally:
                await queue.put(finished)

        # The task is created inside the sink so it inherits the subscription:
        # create_task copies the context at creation, and that is what carries
        # it into the pipeline's own steps.
        sink = (lambda s: queue.put_nowait({"type": "step", **s})) if observed else None
        with step_sink(sink):
            task = asyncio.create_task(pump())
        try:
            while True:
                event = await queue.get()
                if event is finished:
                    break
                # Client disconnects are common on a website widget - a visitor
                # closing the tab mid-answer should stop the work, not log an
                # error.
                if await request.is_disconnected():
                    METRICS.incr("chat.client_disconnected")
                    break
                yield _sse(event)
        finally:
            # A disconnect leaves the pipeline mid-turn. Without this it runs
            # to completion writing into a queue nobody will read.
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            log.info("Stream finished.", **trace.to_dict())
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
