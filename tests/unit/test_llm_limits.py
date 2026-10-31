"""Validates the shared rate-limit state: counters, cooldowns and health stats behave the
same on the in-memory store and the Redis-backed one (run against fakeredis, so no server
is needed), quota reserve/refund/headroom, and the fall-back-when-Redis-is-down wrapper.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import fakeredis
import pytest

from atlas.cognitive.llm.limits import (
    InMemoryRateLimitStore,
    ProviderLimits,
    QuotaTracker,
    RateLimitStore,
    RedisRateLimitStore,
    ResilientRateLimitStore,
)


class FakeClock:
    def __init__(self, now: float = 1_700_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _in_memory(clock: FakeClock) -> RateLimitStore:
    return InMemoryRateLimitStore(now_fn=clock)


def _redis(clock: FakeClock) -> RateLimitStore:
    # fakeredis keys expire on real time; the contract tests below that need expiry use
    # the in-memory store with its fake clock, and exercise Redis TTLs separately.
    return RedisRateLimitStore(fakeredis.FakeRedis(decode_responses=True))


@pytest.fixture(params=["in_memory", "redis"])
def make_store(request: pytest.FixtureRequest) -> Callable[[FakeClock], RateLimitStore]:
    return _in_memory if request.param == "in_memory" else _redis


class TestStoreContract:
    def test_incr_counts_up_and_get_reads_it(
        self, make_store: Callable[[FakeClock], RateLimitStore]
    ) -> None:
        store = make_store(FakeClock())

        assert [store.incr("k", 60) for _ in range(3)] == [1, 2, 3]
        assert store.get("k") == 3
        assert store.get("missing") == 0

    def test_decr_refunds_but_never_goes_negative(
        self, make_store: Callable[[FakeClock], RateLimitStore]
    ) -> None:
        store = make_store(FakeClock())
        store.incr("k", 60)

        store.decr("k")
        store.decr("k")

        assert store.get("k") == 0

    def test_cooldown_is_reported_and_unset_is_zero(
        self, make_store: Callable[[FakeClock], RateLimitStore]
    ) -> None:
        store = make_store(FakeClock())

        assert store.cooldown_remaining("c") == 0.0
        store.set_cooldown("c", 30.0)

        assert 0.0 < store.cooldown_remaining("c") <= 30.0

    def test_stats_default_optimistic_then_track_outcomes(
        self, make_store: Callable[[FakeClock], RateLimitStore]
    ) -> None:
        store = make_store(FakeClock())
        assert store.stats("p").ok_ema == 1.0
        assert store.stats("p").latency_ms_ema is None

        store.record_outcome("p", True, 200.0)
        first = store.stats("p")
        store.record_outcome("p", False, 999.0)  # a failure lowers reliability...
        after_failure = store.stats("p")

        assert first.latency_ms_ema == 200.0
        assert after_failure.ok_ema < first.ok_ema
        assert after_failure.latency_ms_ema == 200.0  # ...but never pollutes latency


class TestHealthStatsRecovery:
    def test_one_failure_dents_reliability_but_does_not_zero_it(self) -> None:
        store = InMemoryRateLimitStore()

        store.record_outcome("p", False, 100.0)

        assert store.stats("p").ok_ema == pytest.approx(0.8)

    def test_in_memory_health_is_forgotten_after_ten_idle_minutes(self) -> None:
        clock = FakeClock()
        store = InMemoryRateLimitStore(now_fn=clock)
        store.record_outcome("p", False, 100.0)
        store.record_outcome("p", False, 100.0)
        assert store.stats("p").ok_ema < 0.7

        clock.now += 601

        assert store.stats("p").ok_ema == 1.0

    def test_redis_health_expires_through_a_ttl(self) -> None:
        client = fakeredis.FakeRedis(decode_responses=True)
        RedisRateLimitStore(client).record_outcome("p", True, 100.0)

        assert 0 < client.ttl("p:stats") <= 600


class TestExpiry:
    def test_in_memory_counters_expire_after_their_ttl(self) -> None:
        clock = FakeClock()
        store = InMemoryRateLimitStore(now_fn=clock)
        store.incr("k", 60)

        clock.now += 61

        assert store.get("k") == 0
        assert store.incr("k", 60) == 1  # a fresh window starts at one

    def test_in_memory_cooldown_expires(self) -> None:
        clock = FakeClock()
        store = InMemoryRateLimitStore(now_fn=clock)
        store.set_cooldown("c", 10.0)

        clock.now += 11

        assert store.cooldown_remaining("c") == 0.0

    def test_redis_counter_gets_a_ttl_atomically_with_the_increment(self) -> None:
        client = fakeredis.FakeRedis(decode_responses=True)
        RedisRateLimitStore(client).incr("k", 120)

        assert 0 < client.ttl("k") <= 120


class TestQuotaTracker:
    LIMITS = ProviderLimits(requests_per_minute=3, requests_per_day=5)

    def test_reserves_up_to_the_minute_cap_then_refuses_without_leaking(self) -> None:
        clock = FakeClock()
        store = InMemoryRateLimitStore(now_fn=clock)
        quota = QuotaTracker(store, now_fn=clock)

        results = [quota.try_reserve("groq", self.LIMITS) for _ in range(5)]

        assert results == [True, True, True, False, False]
        # Refused attempts were refunded: the counters hold exactly what was granted.
        assert quota.headroom("groq", self.LIMITS) == pytest.approx(0.0)

    def test_minute_budget_refills_next_minute_but_day_budget_does_not(self) -> None:
        clock = FakeClock()
        quota = QuotaTracker(InMemoryRateLimitStore(now_fn=clock), now_fn=clock)
        assert all(quota.try_reserve("groq", self.LIMITS) for _ in range(3))
        assert not quota.try_reserve("groq", self.LIMITS)

        clock.now += 61  # next minute bucket

        assert quota.try_reserve("groq", self.LIMITS) is True
        assert quota.try_reserve("groq", self.LIMITS) is True
        # 5 granted so far = the daily cap: the minute is free, the day is not.
        assert quota.try_reserve("groq", self.LIMITS) is False

    def test_headroom_is_the_tighter_of_the_two_budgets(self) -> None:
        clock = FakeClock()
        quota = QuotaTracker(InMemoryRateLimitStore(now_fn=clock), now_fn=clock)
        limits = ProviderLimits(requests_per_minute=10, requests_per_day=4)

        quota.try_reserve("p", limits)
        quota.try_reserve("p", limits)

        # minute: 8/10 left, day: 2/4 left -> the day budget is the bottleneck.
        assert quota.headroom("p", limits) == pytest.approx(0.5)

    def test_providers_have_independent_budgets(self) -> None:
        clock = FakeClock()
        quota = QuotaTracker(InMemoryRateLimitStore(now_fn=clock), now_fn=clock)
        for _ in range(3):
            quota.try_reserve("groq", self.LIMITS)

        assert quota.headroom("groq", self.LIMITS) == 0.0
        assert quota.headroom("gemini", self.LIMITS) == 1.0

    def test_two_trackers_on_one_redis_share_the_budget(self) -> None:
        """The reason Redis is here: separate processes draw from one budget."""
        client = fakeredis.FakeRedis(decode_responses=True)
        clock = FakeClock()
        a = QuotaTracker(RedisRateLimitStore(client), now_fn=clock)
        b = QuotaTracker(RedisRateLimitStore(client), now_fn=clock)

        granted = [a.try_reserve("g", self.LIMITS), b.try_reserve("g", self.LIMITS)]
        granted += [a.try_reserve("g", self.LIMITS), b.try_reserve("g", self.LIMITS)]

        assert granted == [True, True, True, False]

    def test_cooldown_is_shared_through_the_store(self) -> None:
        client = fakeredis.FakeRedis(decode_responses=True)
        a = QuotaTracker(RedisRateLimitStore(client))
        b = QuotaTracker(RedisRateLimitStore(client))

        a.start_cooldown("groq", 20.0)

        assert b.cooldown_remaining("groq") > 0

    def test_prefix_namespaces_the_keys(self) -> None:
        client = fakeredis.FakeRedis(decode_responses=True)
        quota = QuotaTracker(RedisRateLimitStore(client), prefix="test:ns")

        quota.try_reserve("groq", self.LIMITS)

        assert all(k.startswith("test:ns:groq:") for k in client.keys("*"))

    def test_non_positive_limits_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            ProviderLimits(0, 10)


class _BrokenStore:
    """Every operation raises, like a Redis that has gone away."""

    def __getattr__(self, name: str) -> Any:
        def boom(*_: Any) -> Any:
            raise ConnectionError("redis is down")

        return boom


class TestResilientStore:
    def test_falls_back_to_memory_when_the_primary_errors(self) -> None:
        store = ResilientRateLimitStore(_BrokenStore(), InMemoryRateLimitStore())  # type: ignore[arg-type]

        assert store.incr("k", 60) == 1
        assert store.incr("k", 60) == 2
        assert store.get("k") == 2
        store.set_cooldown("c", 5.0)
        assert store.cooldown_remaining("c") > 0
        store.record_outcome("p", True, 100.0)
        assert store.stats("p").latency_ms_ema == 100.0

    def test_uses_the_primary_when_it_works(self) -> None:
        primary = InMemoryRateLimitStore()
        fallback = InMemoryRateLimitStore()
        store = ResilientRateLimitStore(primary, fallback)

        store.incr("k", 60)

        assert primary.get("k") == 1
        assert fallback.get("k") == 0
