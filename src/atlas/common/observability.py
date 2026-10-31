"""Optional error monitoring (Sentry).

Off unless `SENTRY_DSN` is set, and `sentry-sdk` is an optional extra (`--extra observability`),
so nothing here costs anything -- in dependencies, startup time or privacy -- until you opt in.
When on, it is configured conservatively: no PII, no request bodies (operator tokens travel in
headers and incident payloads in bodies), and `release`/`environment` tags so an error can be
tied to the exact image tag and stage that produced it.

The free Sentry tier covers a project this size; any other backend would plug in the same way.
"""

from __future__ import annotations

from collections.abc import Mapping

from atlas.common.logging import get_logger

logger = get_logger(__name__)


def init_sentry(env: Mapping[str, str], release: str, environment: str) -> bool:
    """Initialises Sentry if configured. Returns True only if it was actually enabled; never
    raises (monitoring must not stop the service from starting)."""
    dsn = env.get("SENTRY_DSN", "").strip()
    if not dsn:
        return False
    try:
        import sentry_sdk
    except ImportError:
        logger.warning(
            "sentry.not_installed",
            hint="SENTRY_DSN is set but sentry-sdk is not installed (uv sync --extra observability)",
        )
        return False

    try:
        sentry_sdk.init(
            dsn=dsn,
            release=release,
            environment=environment,
            traces_sample_rate=float(env.get("SENTRY_TRACES_SAMPLE_RATE", "0") or 0),
            send_default_pii=False,
            max_request_body_size="never",
        )
    except Exception:  # a malformed DSN etc.
        logger.exception("sentry.init_failed")
        return False
    logger.info("sentry.enabled", release=release, environment=environment)
    return True
