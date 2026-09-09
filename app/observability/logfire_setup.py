"""Layer 08 - Logfire export.

The system was already instrumented: every step runs inside `tracing.span()`
and every event goes through structlog. This module points both of those at
Logfire, so a run can be *read as a trace* - nested steps with durations and
the model calls underneath them - instead of scrolled as text.

Without a token it is inert. `configure()` still runs, spans are still created,
the terminal still prints, and nothing is sent anywhere. That keeps tests, CI
and an offline laptop on exactly the same code path as an exporting run.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import logfire

from app.config import Settings

_enabled = False
_configured = False

# Span attributes of ours that collide with Logfire's default patterns. The
# scrubber matches on field *name*, and "session" is one of its patterns, so
# `session_id` - a thread key, not a credential - would export as a redaction
# and make the traces unreadable for the one question they are kept to answer.
_OUR_OWN_FIELD_NAMES = frozenset({"session_id", "thread_id"})


def _keep_our_own_field_names(match: logfire.ScrubMatch) -> Any:
    """Suppress the false positives, and let everything else redact.

    Scrubbing stays *on* because `configure()` governs more than `span()`.
    `instrument_fastapi` records request data, headers included, so with the
    scrubber off an `X-Admin-Key` reached the exporter in clear text; the same
    applies to every structlog field forwarded through StructlogProcessor.
    Switching it off wholesale to keep one field name readable was too broad a
    fix for too narrow a problem.

    Note what this does *not* cover: Logfire exempts `logfire.openai` and
    `logfire.anthropic` spans from scrubbing itself, so prompts and completions
    are exported as-is whatever this is set to. Keeping a Visitor's question
    out of Logfire is a decision about whether to call `instrument_openai` at
    all, not one this callback can make.
    """
    if match.path and str(match.path[-1]) in _OUR_OWN_FIELD_NAMES:
        return match.value
    return None


def configure_logfire(settings: Settings, *, service: str) -> bool:
    """Configure the SDK once per process. Returns True if traces are exported.

    `service` separates the CLI from the API in the Logfire UI - both write to
    the same project, and "which process produced this span" is the first
    question you ask of a trace.
    """
    global _enabled, _configured
    if _configured:
        return _enabled

    instance = logfire.configure(
        service_name=service,
        service_version="0.1.0",
        environment=settings.logfire_environment,
        token=settings.logfire_token or None,
        # Only exports when a token is set; no prompt, no failure without one.
        send_to_logfire="if-token-present",
        # structlog owns the terminal (see logging_setup). Leaving this on
        # would print every span twice, in two different formats.
        console=False,
        scrubbing=logfire.ScrubbingOptions(callback=_keep_our_own_field_names),
    )
    _configured = True
    # Not `bool(settings.logfire_token)`: `logfire projects new` writes a token
    # to .logfire/ too, and a run that exports from that file should be
    # instrumented like any other.
    _enabled = bool(instance.config.token)
    return _enabled


def instrument_providers() -> None:
    """Attach spans to the SDKs that cost money, and metrics to the process.

    Each is guarded: an SDK version the integration does not recognise must
    degrade to "no spans for that provider", never to a failed ingest.
    """
    # Imported here, not at module scope: logging_setup imports this module.
    from app.logging_setup import get_logger

    if not _enabled:
        return
    log = get_logger(__name__)
    for name, instrument in (
        ("openai", logfire.instrument_openai),          # answer/rewrite/verify/eval
        ("google_genai", logfire.instrument_google_genai),  # embeddings
        # Not a provider: CPU and memory for the process. Cheap, and the only
        # way to tell "the embedder is slow" from "the box is out of memory".
        ("system_metrics", logfire.instrument_system_metrics),
    ):
        try:
            instrument()
        except Exception as exc:  # noqa: BLE001 - observability is never fatal
            log.warning(f"Could not instrument {name}; its calls will not be traced.",
                        target=name, error=str(exc))


def structlog_processor():
    """Every structlog event, forwarded to Logfire with its fields intact."""
    return logfire.StructlogProcessor()


@contextmanager
def exported_span(name: str, attributes: dict[str, Any]) -> Iterator[Any]:
    """The Logfire half of `tracing.span()`. A no-op when export is off."""
    if not _enabled:
        yield None
        return
    with logfire.span(name, **attributes) as span:
        yield span


def instrument_fastapi(app) -> None:
    """One span per HTTP request, with every pipeline step nested inside it."""
    if _enabled:
        logfire.instrument_fastapi(app, excluded_urls="/health,/guide.js")
