"""Chaos scenario 2/3 (docs/design/11_evaluation.md section 11.5.4): "kill the
preemption overlay mid-run (lock TTL recovery)."

The mechanism itself (SETNX-with-TTL self-expiry against a fake clock) is already
verified in tests/unit/test_lock_store.py, whose own docstrings already call out
"the literal chaos-test scenario" -- this module gives that scenario a dedicated home
under tests/chaos/, narrated as one continuous incident timeline (lock acquired mid-
emergency-run -> overlay process dies without releasing -> TTL elapses -> a fresh run
recovers the junction) with an explicit pass criterion, rather than three separate
unit assertions.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from atlas.coordination.lock_store import InMemoryLockStore

LOCK_TTL_S = 30.0


def test_scenario_overlay_dies_holding_a_lock_junction_recovers_after_ttl() -> None:
    """Pass criterion: a junction lock held by a preemption run that never releases it
    (the overlay process died) must free itself once its TTL elapses, AND a fresh run
    must then be able to acquire the SAME junction -- detection alone (the lock
    eventually shows as unheld) isn't enough; recovery (someone else can actually take
    over) is the real requirement docs/design/04_agent_architecture.md section 4.3
    depends on.
    """
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    now = {"t": t0}
    store = InMemoryLockStore(now_fn=lambda: now["t"])
    junction_id = "42430344"  # a real corridor junction (Week 5/7's own 5-junction corridor)

    # An emergency run acquires the junction's preemption lock, as normal.
    assert store.acquire(junction_id, holder="emergency_run_1", ttl_s=LOCK_TTL_S) is True
    assert store.holder(junction_id) == "emergency_run_1"

    # The overlay process is killed here -- it never calls release(). Time passes,
    # but not yet past the TTL: the lock must still correctly show as held (a
    # premature "free" here would let a second run barge into an ACTIVE preemption).
    now["t"] = t0 + timedelta(seconds=LOCK_TTL_S - 1)
    assert store.holder(junction_id) == "emergency_run_1"

    # Past the TTL: the dead run's lock must have self-expired.
    now["t"] = t0 + timedelta(seconds=LOCK_TTL_S + 1)
    assert store.holder(junction_id) is None

    # Recovery: a fresh run (a new emergency, or a fallback controller reclaiming
    # normal control) must be able to acquire the SAME junction -- not left
    # permanently stranded by the dead overlay's leaked lock.
    assert store.acquire(junction_id, holder="emergency_run_2", ttl_s=LOCK_TTL_S) is True
    assert store.holder(junction_id) == "emergency_run_2"
