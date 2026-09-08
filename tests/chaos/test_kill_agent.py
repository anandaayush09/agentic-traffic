"""Chaos scenario 1/3 (docs/design/11_evaluation.md section 11.5.4): "kill K agent
processes mid-episode (watchdog engagement time, delay cost)."

The mechanism itself (deterministic staleness detection with a fake clock) is already
verified thoroughly in tests/unit/test_watchdog.py -- this module is the scripted
scenario with an explicit pass criterion the design doc's own words ask for, narrated
as the actual incident timeline rather than isolated unit assertions, giving it a
dedicated home under tests/chaos/ instead of leaving the framing implicit.

A real process kill (SIGKILL on a running IntersectionAgentService) is exercised at
the FULL SYSTEM level against a real cluster in scripts/kind_smoke_test.py's own
chaos test (Week 10) -- pod deletion, real k8s self-healing, real reconnect. This
scenario complements that: it's the DETECTION-TIMING guarantee (does staleness fire
within the promised 2-cycle window?), verified deterministically rather than by
racing real wall-clock sleeps against a real kill.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from atlas.services.watchdog import Watchdog

HEARTBEAT_INTERVAL_S = 5.0
STALE_AFTER_S = 2 * HEARTBEAT_INTERVAL_S  # the Week 6 roadmap's own "within 2 cycles"


def test_scenario_agent_process_dies_mid_episode_watchdog_detects_within_two_cycles() -> None:
    """Pass criterion: an agent heartbeating normally, then killed (heartbeats stop
    entirely), is flagged stale by the Watchdog within 2 missed heartbeat cycles --
    neither instantly (a single missed beat is normal jitter, not proof of death) nor
    late (every cycle beyond 2 is added delay before a fallback controller engages).
    """
    watchdog = Watchdog(stale_after_s=STALE_AFTER_S)
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    agent_id = "intersection_agent.42430333"

    # Three normal cycles of the agent alive and heartbeating.
    for cycle in range(3):
        heartbeat_time = t0 + timedelta(seconds=cycle * HEARTBEAT_INTERVAL_S)
        watchdog.record_heartbeat(agent_id, heartbeat_time)
        assert watchdog.stale_services(heartbeat_time) == []

    # The process is killed here (t = 3 * HEARTBEAT_INTERVAL_S) -- no more heartbeats
    # ever arrive for this agent from this point on.
    killed_at = t0 + timedelta(seconds=3 * HEARTBEAT_INTERVAL_S)

    # One missed cycle: not yet stale -- this must not be a false alarm on normal jitter.
    one_cycle_later = killed_at + timedelta(seconds=HEARTBEAT_INTERVAL_S)
    assert watchdog.stale_services(one_cycle_later) == []

    # Two full missed cycles: now genuinely stale -- fallback must be able to engage.
    two_cycles_later = killed_at + timedelta(seconds=STALE_AFTER_S + 0.1)
    assert watchdog.stale_services(two_cycles_later) == [agent_id]

    # Stays stale indefinitely afterward -- it doesn't self-heal just because time
    # keeps passing; only a fresh heartbeat (a real restart) would clear it.
    much_later = killed_at + timedelta(seconds=STALE_AFTER_S * 10)
    assert watchdog.stale_services(much_later) == [agent_id]
