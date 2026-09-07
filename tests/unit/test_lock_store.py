"""Validates InMemoryLockStore's SETNX-with-TTL semantics against a fake clock -- the
actual mechanism the Week 7 "lock-TTL recovery chaos test" needs to be correct, tested
deterministically without real sleeps.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from atlas.coordination.lock_store import InMemoryLockStore

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _clock_at(t: datetime) -> InMemoryLockStore:
    return InMemoryLockStore(now_fn=lambda: t)


class TestAcquireAndHolder:
    def test_first_acquire_succeeds(self) -> None:
        store = _clock_at(T0)
        assert store.acquire("junction_1", holder="run_a", ttl_s=30.0) is True
        assert store.holder("junction_1") == "run_a"

    def test_second_holder_cannot_acquire_a_live_lock(self) -> None:
        store = _clock_at(T0)
        store.acquire("junction_1", holder="run_a", ttl_s=30.0)

        assert store.acquire("junction_1", holder="run_b", ttl_s=30.0) is False
        assert store.holder("junction_1") == "run_a"  # unaffected by the failed attempt

    def test_same_holder_can_reacquire_and_refresh_ttl(self) -> None:
        store = _clock_at(T0)
        store.acquire("junction_1", holder="run_a", ttl_s=10.0)

        assert store.acquire("junction_1", holder="run_a", ttl_s=30.0) is True

    def test_unheld_key_has_no_holder(self) -> None:
        store = _clock_at(T0)
        assert store.holder("never_locked") is None


class TestRelease:
    def test_release_by_actual_holder_frees_the_lock(self) -> None:
        store = _clock_at(T0)
        store.acquire("junction_1", holder="run_a", ttl_s=30.0)

        store.release("junction_1", holder="run_a")

        assert store.holder("junction_1") is None

    def test_release_by_a_different_holder_does_nothing(self) -> None:
        """A stale/late release from a run that already lost this lock must never
        release a DIFFERENT run's live lock."""
        store = _clock_at(T0)
        store.acquire("junction_1", holder="run_a", ttl_s=30.0)

        store.release("junction_1", holder="run_b")

        assert store.holder("junction_1") == "run_a"

    def test_release_of_never_held_key_does_not_raise(self) -> None:
        store = _clock_at(T0)
        store.release("never_locked", holder="run_a")  # must not raise


class TestTtlExpiry:
    """The self-healing property docs/design/04_agent_architecture.md section 4.3
    depends on: a lock leak (overlay process dies before releasing) must not strand a
    junction forever."""

    def test_lock_is_live_before_ttl_elapses(self) -> None:
        now = {"t": T0}
        store = InMemoryLockStore(now_fn=lambda: now["t"])
        store.acquire("junction_1", holder="run_a", ttl_s=30.0)

        now["t"] = T0 + timedelta(seconds=29)

        assert store.holder("junction_1") == "run_a"

    def test_lock_self_expires_after_ttl_with_no_release_call(self) -> None:
        """The literal chaos-test scenario: the overlay 'dies' (never calls release())
        after acquiring -- the lock must still free itself once its TTL elapses."""
        now = {"t": T0}
        store = InMemoryLockStore(now_fn=lambda: now["t"])
        store.acquire("junction_1", holder="run_a", ttl_s=30.0)

        now["t"] = T0 + timedelta(seconds=31)

        assert store.holder("junction_1") is None

    def test_a_different_run_can_acquire_after_expiry(self) -> None:
        """Recovery, not just detection: once the dead overlay's lock expires, a fresh
        run (or a fallback controller) must be able to take over the junction."""
        now = {"t": T0}
        store = InMemoryLockStore(now_fn=lambda: now["t"])
        store.acquire("junction_1", holder="run_a", ttl_s=30.0)
        now["t"] = T0 + timedelta(seconds=31)

        assert store.acquire("junction_1", holder="run_b", ttl_s=30.0) is True
        assert store.holder("junction_1") == "run_b"
