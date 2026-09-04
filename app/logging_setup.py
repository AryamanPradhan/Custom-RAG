"""Structured logging (Layer 08 · Observability).

Every log line carries trace_id and property_id via contextvars so a single
guest request can be reconstructed end to end.
"""

from __future__ import annotations

import logging
import sys

import structlog

_configured = False


def configure_logging(level: str = "INFO") -> None:
    global _configured
    if _configured:
        return

    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level.upper())

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping()[level.upper()]
        ),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str):
    return structlog.get_logger(name)
