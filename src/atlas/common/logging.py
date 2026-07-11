"""Structured logging setup shared by every ATLAS service.

All services log through structlog so that output is machine-parseable (JSON) in
containers/production and human-readable in a local dev console. `bind_context` is used
to attach correlation fields (e.g. `trace_id`, `intersection_id`) to every subsequent log
line emitted on the current execution context -- this is the hook the event-bus layer
(Part 5 of the design docs) will use for per-hop latency tracing once it exists; it is
wired up now so logging call sites do not need to change when that lands.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Literal

import structlog

LogFormat = Literal["console", "json"]

_CONFIGURED = False


def configure_logging(level: str = "INFO", fmt: LogFormat = "console") -> None:
    """Configure structlog for the current process. Safe to call multiple times; only the
    first call takes effect, so importing a module that calls this at import time never
    clobbers a configuration a service entrypoint already set up.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
    )

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    renderer: structlog.types.Processor
    if fmt == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[*shared_processors, structlog.dev.set_exc_info, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _CONFIGURED = True


def get_logger(name: str) -> structlog.typing.FilteringBoundLogger:
    """Return a structlog logger bound to `name` (conventionally the calling module's
    `__name__`). Calls `configure_logging` with defaults if nothing configured it yet, so
    every module can safely call this at import time without an explicit bootstrap step.
    """
    if not _CONFIGURED:
        configure_logging()
    logger: structlog.typing.FilteringBoundLogger = structlog.get_logger(name)
    return logger


def bind_context(**kwargs: Any) -> None:
    """Bind key/value pairs to every log line emitted from the current context (thread /
    asyncio task) until `clear_context` is called. Used to attach correlation ids such as
    `trace_id` or `intersection_id` without threading them through every function call.
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context() -> None:
    """Clear all context-bound log fields for the current execution context."""
    structlog.contextvars.clear_contextvars()
