"""Structured logging (Layer 08 · Observability).

A run reads as a timeline of steps - one line each, in the order they ran:

    18:35:03  🧠 Planner Decision    1.13s  Rewrote the question into 2 queries.
    18:35:07  🔎 Vector Search       3.67s  Retrieved 10 candidates from Qdrant.
    18:35:07  🎯 Reranking            161ms  Reranked 10 candidates down to 4.

The message is a sentence because a person reads it; the fields behind it stay
structured because Logfire and `LOG_FORMAT=json` index them. Every line also
carries trace_id and property_id via contextvars, so one guest request can be
reconstructed end to end.
"""

from __future__ import annotations

import logging
import sys
import unicodedata

import structlog

from app.config import Settings, get_settings
from app.observability.logfire_setup import (
    configure_logfire,
    instrument_providers,
    structlog_processor,
)

_configured = False


def configure_logging(level: str = "INFO", *, service: str = "guide") -> None:
    """Configure structlog and, if a token is set, the Logfire exporter.

    Called once at the start of every entry point - the CLI and the API
    lifespan - and a no-op after that.
    """
    global _configured
    if _configured:
        return

    settings = get_settings()
    exporting = configure_logfire(settings, service=service)

    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level.upper())
    _quieten_dependencies()

    processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        _timestamper(settings),
        structlog.processors.StackInfoRenderer(),
    ]
    if exporting:
        # Before format_exc_info, which consumes exc_info into a string: the
        # exporter wants the exception object so Logfire gets a real traceback.
        processors.append(structlog_processor())
    processors += [
        structlog.processors.format_exc_info,
        _renderer(settings),
    ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping()[level.upper()]
        ),
        cache_logger_on_first_use=True,
    )
    _configured = True
    # After structlog is configured, so that a failure here is reported in the
    # same format as every other line.
    instrument_providers()


# Libraries that log one INFO line per HTTP request or per downloaded file.
# At our INFO level they bury the step timeline under hundreds of lines about
# redirects and cache hits - all of it a level below what an operator wants.
_NOISY = (
    "httpx",
    "httpcore",
    # The OpenAI SDK ships its own fork, so silencing "httpx" alone leaves one
    # "HTTP Request: POST ..." line per model call.
    "httpx2",
    "httpcore2",
    "huggingface_hub",
    "filelock",
    "urllib3",
    "fsspec",
    "openai",
    "asyncio",
)


def _quieten_dependencies() -> None:
    for name in _NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)


def _timestamper(settings: Settings):
    # Wall-clock time is what you correlate against in the terminal; the full
    # ISO stamp is what a shipper needs.
    if settings.log_format == "console":
        return structlog.processors.TimeStamper(fmt="%H:%M:%S")
    return structlog.processors.TimeStamper(fmt="iso", utc=True)


def _renderer(settings: Settings):
    if settings.log_format == "console":
        return StepRenderer(colors=sys.stdout.isatty(), icons=_unicode_stdout())
    return structlog.processors.JSONRenderer()


_DIM = "\x1b[2m"
_RED = "\x1b[31m"
_YELLOW = "\x1b[33m"
_BOLD = "\x1b[1m"
_RESET = "\x1b[0m"

# Bound to the request, not to the step. Printing them on every line would
# treble its length to repeat what the run as a whole already established.
_CONTEXT_KEYS = {"trace_id", "session_id", "property_id"}


class StepRenderer:
    """A run as a timeline of steps.

        18:35:03  🧠 Planner Decision      1.13s  Rewrote into 2 queries.
        18:35:07  🔎 Vector Search         3.67s  Retrieved 10 candidates.
        18:35:07  🎯 Reranking              161ms  Reranked down to 4 documents.

    Fields are not printed when the step already said what happened in words -
    they are still exported to Logfire and still present under LOG_FORMAT=json.
    A line without a sentence, and any warning or error, prints them.
    """

    LABEL_WIDTH = 20

    def __init__(self, *, colors: bool, icons: bool) -> None:
        self._colors = colors
        self._icons = icons

    def __call__(self, logger, name, event_dict) -> str:
        timestamp = event_dict.pop("timestamp", "")
        level = event_dict.pop("level", "info")
        icon = event_dict.pop("icon", "")
        step = event_dict.pop("step", "")
        ms = event_dict.pop("ms", None)
        message = event_dict.pop("event", "") or ""
        exception = event_dict.pop("exception", None)
        event_dict.pop("logger", None)

        head = f"{icon} {step}" if (step and self._icons) else step
        columns = [
            self._paint(timestamp, _DIM),
            self._pad(head, self.LABEL_WIDTH),
            self._paint(f"{_duration(ms):>7}", _DIM),
            self._paint(message, _RED if level == "error" else
                        _YELLOW if level == "warning" else ""),
        ]
        line = "  ".join(columns).rstrip()

        fields = {k: v for k, v in event_dict.items() if k not in _CONTEXT_KEYS}
        if fields and (not message or level in ("warning", "error")):
            rendered = " ".join(f"{k}={_short(v)}" for k, v in sorted(fields.items()))
            line = f"{line}  {self._paint(rendered, _DIM)}"
        if exception:
            line = f"{line}\n{exception}"
        return line

    def _paint(self, text: str, colour: str) -> str:
        return f"{colour}{text}{_RESET}" if (self._colors and colour and text) else text

    def _pad(self, text: str, width: int) -> str:
        padding = max(0, width - _display_width(text))
        return self._paint(text, _BOLD) + " " * padding


def _duration(ms: float | None) -> str:
    if ms is None:
        return ""
    return f"{ms:.0f}ms" if ms < 1000 else f"{ms / 1000:.2f}s"


def _short(value, limit: int = 40) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _display_width(text: str) -> int:
    """Terminal columns, not codepoints - an emoji occupies two of them.

    `east_asian_width` alone is not enough: it calls ✂ and 🗂 narrow, yet both
    render double-width once a variation selector asks for the emoji form.
    """
    width = 0
    for i, char in enumerate(text):
        if char == "\ufe0f":  # variation selector: no width of its own
            continue
        emoji = (
            unicodedata.east_asian_width(char) in ("W", "F")
            or ord(char) >= 0x1F300
            or text[i + 1 : i + 2] == "\ufe0f"
        )
        width += 2 if emoji else 1
    return width


def _unicode_stdout() -> bool:
    """True if step icons are printable. Windows pipes default to cp1252."""
    for _ in range(2):
        try:
            "\u2705\U0001f9e0".encode(sys.stdout.encoding or "ascii")
            return True
        except (LookupError, UnicodeEncodeError):
            try:
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            except Exception:  # noqa: BLE001 - a non-reconfigurable stream
                return False
    return False


def get_logger(name: str):
    return structlog.get_logger(name)
