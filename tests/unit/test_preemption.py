"""Validates PreemptionOverlay's PLANNED -> LOCKING -> ACTIVE -> RELEASING -> done state
machine, plus the two Week 7 chaos-test scenarios the roadmap explicitly asks for:
lock-TTL recovery (kill the overlay mid-run) and a starvation bound for conflicting
emergency routes. All driven by explicit `now_s` values, no real sleeps or process
control needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import pytest

from atlas.bus.producer import BusProducer
from atlas.bus.schema_registry import SchemaRegistry
from atlas.coordination.lock_store import InMemoryLockStore
from atlas.coordination.preemption import PreemptionOverlay, RouteStop

T0 = datetime(2026, 1, 1, tzinfo=UTC)
CLEARANCE_S = 15.0


def _overlay(now_fn=None, bus_producer=None) -> PreemptionOverlay:  # type: ignore[no-untyped-def]
    return PreemptionOverlay(
        InMemoryLockStore(now_fn=now_fn), clearance_s=CLEARANCE_S, bus_producer=bus_producer
    )


@dataclass
class _FakeRawProducer:
    sent: list[tuple[str, bytes | None, bytes | None]] = field(default_factory=list)

    def send(self, topic: str, value: bytes | None = None, key: bytes | None = None) -> object:
        self.sent.append((topic, value, key))
        return None

    def flush(self, timeout: float | None = None) -> None:
        pass


def _route() -> list[RouteStop]:
    return [
        RouteStop(tls_id="J1", phase_id=0, eta_s=20.0),
        RouteStop(tls_id="J2", phase_id=1, eta_s=50.0),
    ]


class TestSingleRunLifecycle:
    def test_starts_planned(self) -> None:
        overlay = _overlay()
        run = overlay.trigger("run_1", _route())
        assert run.state == "planned"

    def test_enters_locking_once_within_clearance_window_of_first_stop(self) -> None:
        overlay = _overlay()
        overlay.trigger("run_1", _route())

        run = overlay.advance("run_1", now_s=20.0 - CLEARANCE_S)  # exactly at window start

        assert run.state in ("locking", "active")  # acquire succeeds immediately -> active
        assert "J1" in run.locks_acquired

    def test_becomes_active_once_a_lock_is_acquired(self) -> None:
        overlay = _overlay()
        overlay.trigger("run_1", _route())

        run = overlay.advance("run_1", now_s=10.0)  # within J1's window (20 +/- 15)

        assert run.state == "active"
        assert overlay.active_phase_for("J1") == 0

    def test_releases_after_the_junction_window_ends(self) -> None:
        overlay = _overlay()
        overlay.trigger("run_1", _route())
        overlay.advance("run_1", now_s=10.0)  # acquire J1

        run = overlay.advance("run_1", now_s=20.0 + CLEARANCE_S + 0.1)  # past J1's window

        assert "J1" in run.locks_released
        assert overlay.active_phase_for("J1") is None

    def test_reaches_done_once_every_stop_is_released(self) -> None:
        overlay = _overlay()
        overlay.trigger("run_1", _route())
        overlay.advance("run_1", now_s=10.0)  # acquire J1
        overlay.advance("run_1", now_s=36.0)  # release J1, acquire J2
        run = overlay.advance("run_1", now_s=66.0)  # release J2

        assert run.state == "done"
        assert run.locks_released == {"J1", "J2"}

    def test_advance_after_done_is_a_no_op(self) -> None:
        overlay = _overlay()
        overlay.trigger("run_1", _route())
        overlay.advance("run_1", now_s=10.0)
        overlay.advance("run_1", now_s=36.0)
        overlay.advance("run_1", now_s=66.0)

        run = overlay.advance("run_1", now_s=1000.0)  # must not raise or change state

        assert run.state == "done"

    def test_rejects_empty_route(self) -> None:
        overlay = _overlay()
        with pytest.raises(ValueError, match="non-empty"):
            overlay.trigger("run_1", [])


class TestBusPublishing:
    def test_publishes_a_valid_lock_acquired_event(self) -> None:
        raw = _FakeRawProducer()
        bus = BusProducer(raw, SchemaRegistry())
        overlay = _overlay(bus_producer=bus)
        overlay.trigger("run_1", _route())

        overlay.advance("run_1", now_s=10.0)  # acquires J1

        topics = [topic for topic, _, _ in raw.sent]
        assert "control.preemption.v1" in topics

    def test_publishes_a_valid_lock_denied_event(self) -> None:
        raw = _FakeRawProducer()
        bus = BusProducer(raw, SchemaRegistry())
        overlay = _overlay(bus_producer=bus)
        overlay.trigger("run_a", [RouteStop(tls_id="J1", phase_id=0, eta_s=10.0)])
        overlay.trigger("run_b", [RouteStop(tls_id="J1", phase_id=1, eta_s=10.0)])
        overlay.advance("run_a", now_s=5.0)

        overlay.advance("run_b", now_s=5.0)  # must be denied, and denial must still validate

        assert len(raw.sent) == 2  # run_a's acquire + run_b's denial, both schema-valid


class TestLockTtlRecoveryChaosTest:
    """The literal Week 7 ask: 'kill overlay mid-run, assert fallback engages within 2
    cycles'. Simulated by never calling advance() again after a lock is acquired (the
    process 'dying') and checking the lock self-expires -- the actual mechanism a real
    process kill would exercise, verified deterministically."""

    def test_junction_returns_to_normal_control_if_overlay_dies_holding_a_lock(self) -> None:
        now = {"t": T0}
        overlay = _overlay(now_fn=lambda: now["t"])
        overlay.trigger("run_1", _route())
        overlay.advance("run_1", now_s=10.0)  # acquires J1, overlay "alive" here
        assert overlay.active_phase_for("J1") == 0

        # The overlay process dies right here -- no more advance() calls ever happen for
        # this run. Simulate real time passing past the lock's TTL (2 * clearance_s).
        now["t"] = T0 + timedelta(seconds=2 * CLEARANCE_S + 1)

        assert overlay.active_phase_for("J1") is None  # fallback control is safe to resume

    def test_a_fresh_run_can_claim_the_junction_after_the_dead_overlays_lock_expires(
        self,
    ) -> None:
        now = {"t": T0}
        overlay = _overlay(now_fn=lambda: now["t"])
        overlay.trigger("dead_run", [RouteStop(tls_id="J1", phase_id=0, eta_s=10.0)])
        overlay.advance("dead_run", now_s=5.0)  # acquires J1
        now["t"] = T0 + timedelta(seconds=2 * CLEARANCE_S + 1)  # dead_run's lock expires

        overlay.trigger("new_run", [RouteStop(tls_id="J1", phase_id=1, eta_s=5.0)])
        run = overlay.advance("new_run", now_s=5.0)

        assert "J1" in run.locks_acquired
        assert overlay.active_phase_for("J1") == 1


class TestStarvationBoundForConflictingRoutes:
    """The literal Week 7 ask: a starvation-bound test for conflicting emergency
    routes -- first-lock-wins per junction (docs/design/04_agent_architecture.md
    section 4.3), and the LOSING run must be bounded (it doesn't hang forever waiting;
    it's marked denied and the rest of its route proceeds unaffected)."""

    def test_second_run_to_reach_a_contested_junction_is_denied_not_blocked_forever(
        self,
    ) -> None:
        overlay = _overlay()
        overlay.trigger("run_a", [RouteStop(tls_id="J1", phase_id=0, eta_s=10.0)])
        overlay.trigger("run_b", [RouteStop(tls_id="J1", phase_id=1, eta_s=10.0)])

        run_a = overlay.advance("run_a", now_s=5.0)
        run_b = overlay.advance("run_b", now_s=5.0)  # same junction, same window

        assert "J1" in run_a.locks_acquired
        assert "J1" in run_b.locks_denied
        assert overlay.active_phase_for("J1") == 0  # run_a's phase wins, not run_b's

    def test_losing_run_still_proceeds_on_its_other_junctions(self) -> None:
        """A denied lock at one junction must not stall the rest of that run's route --
        the actual 'bound' in starvation-bound: losing one contested junction costs
        exactly that junction, not the whole run."""
        overlay = _overlay()
        overlay.trigger("run_a", [RouteStop(tls_id="J1", phase_id=0, eta_s=10.0)])
        overlay.trigger(
            "run_b",
            [
                RouteStop(tls_id="J1", phase_id=1, eta_s=10.0),
                RouteStop(tls_id="J2", phase_id=0, eta_s=40.0),
            ],
        )
        overlay.advance("run_a", now_s=5.0)
        overlay.advance("run_b", now_s=5.0)  # run_b denied J1

        run_b = overlay.advance("run_b", now_s=30.0)  # J2's window opens independently

        assert "J1" in run_b.locks_denied
        assert "J2" in run_b.locks_acquired
        assert run_b.state == "active"

    def test_denied_lock_can_still_be_won_later_if_it_frees_up(self) -> None:
        """Not a permanent denial -- if the winning run releases (or its lock expires)
        before the losing run's window closes, the loser can still acquire it on a
        later advance() call. Bounded starvation, not indefinite exclusion."""
        overlay = _overlay()
        overlay.trigger("run_a", [RouteStop(tls_id="J1", phase_id=0, eta_s=10.0)])
        overlay.trigger("run_b", [RouteStop(tls_id="J1", phase_id=1, eta_s=10.0)])
        overlay.advance("run_a", now_s=5.0)  # run_a wins
        overlay.advance("run_b", now_s=5.0)  # run_b denied

        overlay.advance("run_a", now_s=10.0 + CLEARANCE_S + 0.1)  # run_a releases J1
        run_b = overlay.advance("run_b", now_s=10.0 + CLEARANCE_S + 0.2)

        assert "J1" in run_b.locks_acquired
