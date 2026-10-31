"""Shared rate-limit and health state for the LLM providers.

Free-tier LLM APIs enforce requests-per-minute and requests-per-day caps (and a 429 when
you exceed them). With two providers (Groq and Gemini) and possibly several processes
(the API, the evaluation scripts), the counters have to be shared and atomic -- that is
what Redis is for here. `RateLimitStore` is the swap point, the same pattern as
`atlas.coordination.lock_store.LockStore`:

- `RedisRateLimitStore`: shared across processes, survives restarts.
- `InMemoryRateLimitStore`: process-local; used in tests and as the automatic fallback
  when Redis is unreachable (`ResilientRateLimitStore`), so a Redis outage degrades
  rate-limit accuracy, never the incident flow.

What is tracked per provider:

- **request counters** in fixed time buckets (minute, UTC day). A request is *reserved*
  before the call (increment, refund if over the cap) so concurrent callers can't all slip
  under a cap together.
- **cooldown**: after a 429 (honouring `Retry-After`), an auth failure (a bad key will not
  fix itself), or another failure, the provider is skipped until the cooldown expires --
  failing fast beats hammering an API that has already said no.
- **health stats**: exponential moving averages of success rate and latency, used by the
  router to prefer the provider that is actually working and fast. They start optimistic
  (1.0) and expire after ten idle minutes, so a provider that failed is retried and can
  recover rather than being starved by its own bad record.

Limits themselves are configuration, not constants: free-tier numbers differ by model and
change over time, so the defaults are conservative and overridable by environment.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from atlas.common.logging import get_logger

logger = get_logger(__name__)

_MINUTE_TTL_S = 120
_DAY_TTL_S = 2 * 24 * 3600
_EMA_ALPHA = 0.2
# Health evidence is forgotten after this long without a new sample. Without it, one bad
# patch would rank a provider last for good: nothing routes to it, so nothing could ever
# show it has recovered.
_STATS_TTL_S = 600


@dataclass(frozen=True, slots=True)
class ProviderLimits:
    requests_per_minute: int
    requests_per_day: int

    def __post_init__(self) -> None:
        if self.requests_per_minute <= 0 or self.requests_per_day <= 0:
            raise ValueError("request limits must be positive")


@dataclass(frozen=True, slots=True)
class ProviderStats:
    ok_ema: float = 1.0
    """Moving average success rate in 0..1 (optimistic 1.0 until we have evidence)."""
    latency_ms_ema: float | None = None


class RateLimitStore(Protocol):
    def incr(self, key: str, ttl_s: int) -> int: ...
    def decr(self, key: str) -> None: ...
    def get(self, key: str) -> int: ...
    def set_cooldown(self, key: str, seconds: float) -> None: ...
    def cooldown_remaining(self, key: str) -> float: ...
    def record_outcome(self, provider_key: str, ok: bool, latency_ms: float) -> None: ...
    def stats(self, provider_key: str) -> ProviderStats: ...


def _next_ema(previous: float | None, sample: float) -> float:
    return sample if previous is None else previous + _EMA_ALPHA * (sample - previous)


class InMemoryRateLimitStore:
    """Process-local store. Correct within one process; `now_fn` is injectable so
    expiry and cooldowns are testable with a fake clock, not real sleeps."""

    def __init__(self, now_fn: Callable[[], float] | None = None) -> None:
        self._now_fn = now_fn or time.time
        self._counters: dict[str, tuple[int, float]] = {}  # key -> (count, expires_at)
        self._cooldowns: dict[str, float] = {}  # key -> expires_at
        self._stats: dict[str, tuple[float | None, float | None, float]] = {}

    def incr(self, key: str, ttl_s: int) -> int:
        now = self._now_fn()
        count, expires_at = self._counters.get(key, (0, 0.0))
        if expires_at <= now:
            count = 0
        count += 1
        self._counters[key] = (count, now + ttl_s)
        return count

    def decr(self, key: str) -> None:
        count, expires_at = self._counters.get(key, (0, 0.0))
        if count > 0:
            self._counters[key] = (count - 1, expires_at)

    def get(self, key: str) -> int:
        count, expires_at = self._counters.get(key, (0, 0.0))
        return count if expires_at > self._now_fn() else 0

    def set_cooldown(self, key: str, seconds: float) -> None:
        self._cooldowns[key] = self._now_fn() + seconds

    def cooldown_remaining(self, key: str) -> float:
        return max(0.0, self._cooldowns.get(key, 0.0) - self._now_fn())

    def record_outcome(self, provider_key: str, ok: bool, latency_ms: float) -> None:
        current = self.stats(provider_key)
        self._stats[provider_key] = (
            _next_ema(current.ok_ema, 1.0 if ok else 0.0),
            _next_ema(current.latency_ms_ema, latency_ms) if ok else current.latency_ms_ema,
            self._now_fn() + _STATS_TTL_S,
        )

    def stats(self, provider_key: str) -> ProviderStats:
        ok_ema, latency_ema, expires_at = self._stats.get(provider_key, (None, None, 0.0))
        if expires_at <= self._now_fn():
            return ProviderStats()
        return ProviderStats(ok_ema=1.0 if ok_ema is None else ok_ema, latency_ms_ema=latency_ema)


class RedisRateLimitStore:
    """Redis-backed store (`redis-py`). `client` is any `redis.Redis`-compatible object,
    so tests run it against `fakeredis` with no server."""

    def __init__(self, client: Any) -> None:
        self._r = client

    def incr(self, key: str, ttl_s: int) -> int:
        # One MULTI/EXEC round trip: the counter and its expiry are set atomically, so a
        # crash between the two can't leave an immortal counter.
        pipe = self._r.pipeline(transaction=True)
        pipe.incr(key)
        pipe.expire(key, ttl_s)
        count, _ = pipe.execute()
        return int(count)

    def decr(self, key: str) -> None:
        if int(self._r.get(key) or 0) > 0:
            self._r.decr(key)

    def get(self, key: str) -> int:
        return int(self._r.get(key) or 0)

    def set_cooldown(self, key: str, seconds: float) -> None:
        self._r.set(key, 1, px=max(1, int(seconds * 1000)))

    def cooldown_remaining(self, key: str) -> float:
        ttl_ms = int(self._r.pttl(key))
        return max(0.0, ttl_ms / 1000.0)  # -2 (no key) / -1 (no expiry) -> 0

    def record_outcome(self, provider_key: str, ok: bool, latency_ms: float) -> None:
        # Advisory statistics: a read-modify-write race between processes just blurs an
        # average slightly, so no transaction is warranted.
        key = f"{provider_key}:stats"
        current = self.stats(provider_key)
        mapping = {"ok_ema": _next_ema(current.ok_ema, 1.0 if ok else 0.0)}
        if ok:
            mapping["latency_ms_ema"] = _next_ema(current.latency_ms_ema, latency_ms)
        pipe = self._r.pipeline(transaction=True)
        pipe.hset(key, mapping=mapping)
        pipe.expire(key, _STATS_TTL_S)
        pipe.execute()

    def stats(self, provider_key: str) -> ProviderStats:
        raw = self._r.hmget(f"{provider_key}:stats", ["ok_ema", "latency_ms_ema"])
        return ProviderStats(
            ok_ema=float(raw[0]) if raw[0] is not None else 1.0,
            latency_ms_ema=float(raw[1]) if raw[1] is not None else None,
        )


class ResilientRateLimitStore:
    """Uses `primary` (Redis) until it errors, then falls back to `fallback` (in-memory)
    for that call and logs once -- a Redis outage must never take incident handling
    down with it. It retries the primary on every call, so recovery is automatic."""

    def __init__(self, primary: RateLimitStore, fallback: RateLimitStore) -> None:
        self._primary = primary
        self._fallback = fallback
        self._warned = False

    def _call(self, name: str, *args: Any) -> Any:
        try:
            result = getattr(self._primary, name)(*args)
            self._warned = False
            return result
        except Exception as exc:  # redis.RedisError and anything socket-level
            if not self._warned:
                logger.warning("rate_limit_store.primary_unavailable", error=str(exc))
                self._warned = True
            return getattr(self._fallback, name)(*args)

    def incr(self, key: str, ttl_s: int) -> int:
        return int(self._call("incr", key, ttl_s))

    def decr(self, key: str) -> None:
        self._call("decr", key)

    def get(self, key: str) -> int:
        return int(self._call("get", key))

    def set_cooldown(self, key: str, seconds: float) -> None:
        self._call("set_cooldown", key, seconds)

    def cooldown_remaining(self, key: str) -> float:
        return float(self._call("cooldown_remaining", key))

    def record_outcome(self, provider_key: str, ok: bool, latency_ms: float) -> None:
        self._call("record_outcome", provider_key, ok, latency_ms)

    def stats(self, provider_key: str) -> ProviderStats:
        result: ProviderStats = self._call("stats", provider_key)
        return result


class QuotaTracker:
    """Provider-level quota logic over a `RateLimitStore`: reserve/refund requests
    against the minute and day caps, cooldowns, and a 0..1 headroom measure."""

    def __init__(
        self,
        store: RateLimitStore,
        now_fn: Callable[[], float] | None = None,
        prefix: str = "atlas:llm",
    ) -> None:
        self._store = store
        self._now_fn = now_fn or time.time
        self._prefix = prefix

    def provider_key(self, provider: str) -> str:
        return f"{self._prefix}:{provider}"

    def _minute_key(self, provider: str) -> str:
        return f"{self.provider_key(provider)}:rpm:{int(self._now_fn() // 60)}"

    def _day_key(self, provider: str) -> str:
        day = datetime.fromtimestamp(self._now_fn(), UTC).strftime("%Y%m%d")
        return f"{self.provider_key(provider)}:rpd:{day}"

    def _cooldown_key(self, provider: str) -> str:
        return f"{self.provider_key(provider)}:cooldown"

    def try_reserve(self, provider: str, limits: ProviderLimits) -> bool:
        """Takes one request from both the minute and day budgets, or takes nothing and
        returns False. Increment-then-refund (not check-then-increment) so two callers
        racing for the last slot can't both get it."""
        minute_key, day_key = self._minute_key(provider), self._day_key(provider)
        minute_count = self._store.incr(minute_key, _MINUTE_TTL_S)
        day_count = self._store.incr(day_key, _DAY_TTL_S)
        if minute_count > limits.requests_per_minute or day_count > limits.requests_per_day:
            self._store.decr(minute_key)
            self._store.decr(day_key)
            return False
        return True

    def headroom(self, provider: str, limits: ProviderLimits) -> float:
        """Fraction (0..1) of the tighter of the minute/day budgets still unused."""
        minute_used = self._store.get(self._minute_key(provider))
        day_used = self._store.get(self._day_key(provider))
        minute_left = max(0.0, 1.0 - minute_used / limits.requests_per_minute)
        day_left = max(0.0, 1.0 - day_used / limits.requests_per_day)
        return min(minute_left, day_left)

    def cooldown_remaining(self, provider: str) -> float:
        return self._store.cooldown_remaining(self._cooldown_key(provider))

    def start_cooldown(self, provider: str, seconds: float) -> None:
        self._store.set_cooldown(self._cooldown_key(provider), seconds)

    def record_outcome(self, provider: str, ok: bool, latency_ms: float) -> None:
        self._store.record_outcome(self.provider_key(provider), ok, latency_ms)

    def stats(self, provider: str) -> ProviderStats:
        return self._store.stats(self.provider_key(provider))
