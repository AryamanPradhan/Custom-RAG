"""Layer 08 - Tracing.

A request trace is the list of steps the agent took: which guardrails fired,
what was searched, what was retrieved, which tools ran, how long each took.
Held in a contextvar so any layer can append without threading a span object
through every signature.

Traces are what make an agentic RAG debuggable. Without them "the answer was
wrong" is unactionable; with them you can see whether retrieval missed, the
reranker dropped the right chunk, or the model ignored it.
"""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

import structlog

from app.logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class Span:
    name: str
    started_at: float
    duration_ms: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class Trace:
    trace_id: str
    property_id: str
    session_id: str
    spans: list[Span] = field(default_factory=list)
    started_at: float = field(default_factory=time.perf_counter)

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "property_id": self.property_id,
            "session_id": self.session_id,
            "total_ms": round((time.perf_counter() - self.started_at) * 1000, 1),
            "spans": [
                {
                    "name": s.name,
                    "duration_ms": s.duration_ms,
                    "error": s.error,
                    **s.attributes,
                }
                for s in self.spans
            ],
        }


_current: ContextVar[Trace | None] = ContextVar("current_trace", default=None)


def start_trace(property_id: str, session_id: str) -> Trace:
    trace = Trace(trace_id=uuid.uuid4().hex[:16], property_id=property_id, session_id=session_id)
    _current.set(trace)
    structlog.contextvars.bind_contextvars(
        trace_id=trace.trace_id, property_id=property_id, session_id=session_id
    )
    return trace


def current_trace() -> Trace | None:
    return _current.get()


def current_trace_id() -> str:
    trace = _current.get()
    return trace.trace_id if trace else "-"


def end_trace() -> None:
    structlog.contextvars.clear_contextvars()
    _current.set(None)


@contextmanager
def span(name: str, **attributes: Any):
    """Time a step and attach it to the active trace.

    Yields the Span so the body can attach results discovered mid-flight:

        with span("retrieve") as s:
            hits = search(...)
            s.attributes["hits"] = len(hits)
    """
    s = Span(name=name, started_at=time.perf_counter(), attributes=dict(attributes))
    trace = _current.get()
    if trace is not None:
        trace.spans.append(s)
    try:
        yield s
    except Exception as exc:
        s.error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        s.duration_ms = round((time.perf_counter() - s.started_at) * 1000, 1)
        log.debug("span", name=name, duration_ms=s.duration_ms, **s.attributes)
