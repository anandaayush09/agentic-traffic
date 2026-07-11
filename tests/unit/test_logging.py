from __future__ import annotations

from atlas.common.logging import bind_context, clear_context, get_logger


def test_get_logger_returns_usable_logger() -> None:
    logger = get_logger(__name__)

    # Should not raise; structlog loggers accept arbitrary keyword event fields.
    logger.info("test.event", example_field=1)


def test_bind_and_clear_context_do_not_raise() -> None:
    bind_context(trace_id="abc-123")
    try:
        logger = get_logger(__name__)
        logger.info("test.context_event")
    finally:
        clear_context()
