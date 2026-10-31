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


class TestLogFormatFromEnvironment:
    """`ATLAS_LOG_FORMAT=json` is what containers set so logs are searchable. It must win even
    though some module always asks for a logger (and thus triggers the default configuration)
    before the service entrypoint runs -- the bug this guards against."""

    def _run(self, env_overrides: dict[str, str]) -> str:
        import os
        import subprocess
        import sys
        from pathlib import Path

        src = Path(__file__).resolve().parents[2] / "src"
        code = (
            "from atlas.common.logging import get_logger\n"
            "get_logger('t').info('hello.event', field=1)\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(src.parent / "tests"),  # not the repo root: don't pick up a developer's .env
            env={**os.environ, "PYTHONPATH": str(src), **env_overrides},
        )
        assert result.returncode == 0, result.stderr[-300:]
        return result.stdout

    def test_json_format_is_honoured_by_the_default_configuration(self) -> None:
        import json

        out = self._run({"ATLAS_LOG_FORMAT": "json"})

        record = json.loads(out.strip().splitlines()[-1])
        assert record["event"] == "hello.event"
        assert record["field"] == 1
        assert record["level"] == "info"

    def test_console_format_is_the_default(self) -> None:
        out = self._run({"ATLAS_LOG_FORMAT": "console"})

        assert "hello.event" in out
        assert not out.strip().startswith("{")

    def test_log_level_is_honoured(self) -> None:
        out = self._run({"ATLAS_LOG_FORMAT": "json", "ATLAS_LOG_LEVEL": "ERROR"})

        assert "hello.event" not in out  # an info event is filtered at ERROR

    def test_an_invalid_setting_falls_back_instead_of_breaking_logging(self) -> None:
        out = self._run({"ATLAS_LOG_FORMAT": "xml"})

        assert "hello.event" in out
