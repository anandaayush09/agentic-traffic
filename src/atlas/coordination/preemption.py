"""Emergency Preemption Overlay: the deterministic state machine that acquires locks
junction-by-junction just-in-time along an emergency vehicle's route, commands green
waves, and releases locks behind the vehicle (docs/design/04_agent_architecture.md
section 4.3). Deliberately NOT learned -- "preemption is safety-critical, rare, and
already solved; learning it would be negligent engineering" (that section's own words).

Driven by an explicit `now_s` (seconds since the run was triggered) rather than wall-
clock time or a background thread, so the entire PLANNED -> LOCKING -> ACTIVE ->
RELEASING -> done lifecycle -- including both Week 7 chaos-test scenarios (lock-TTL
recovery, starvation between conflicting routes) -- is deterministic and testable by
just calling `advance()` with chosen values, no real sleeps or process control needed.
Phase selection (which green phase serves the emergency approach at each junction) is
supplied by the caller as part of the route, not derived here -- the overlay's job is
locking and timing, not geometry; deriving "which phase faces which approach" from the
network is a separate, later concern (the same scoping choice already made for
Webster's formula and green-wave distances: the algorithm itself first, wired to full
live geometry once that's needed).

Two deliberately DIFFERENT clocks, not an inconsistency: `advance(run_id, now_s)`'s
`now_s` is simulation time (seconds since the run was triggered, which may run faster or
slower than real time under libsumo) -- it decides WHICH junction to lock/release based
on where the emergency vehicle logically is. The `LockStore`'s TTL, in contrast, is
always real wall-clock time (via its own `now_fn`, defaulting to `datetime.now`) --
"the overlay process died" is a real-world event, not a simulation event, so the safety-
net that protects against it must be measured in real time regardless of how fast the
simulation clock is running. Conflating the two would make the TTL meaningless as an
actual process-liveness guarantee.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from atlas.bus.producer import BusProducer
from atlas.coordination.lock_store import LockStore

RunState = Literal["planned", "locking", "active", "releasing", "done"]
Priority = Literal["emergency", "vip"]


@dataclass(frozen=True, slots=True)
class RouteStop:
    tls_id: str
    phase_id: int
    eta_s: float
    """Seconds from run start until the vehicle reaches this junction."""


@dataclass(slots=True)
class EmergencyRun:
    run_id: str
    route: tuple[RouteStop, ...]
    priority: Priority
    state: RunState = "planned"
    locks_acquired: set[str] = field(default_factory=set)
    locks_released: set[str] = field(default_factory=set)
    locks_denied: set[str] = field(default_factory=set)
    """Junctions this run tried to lock but lost to a higher-priority or earlier-
    arriving conflicting run -- the observable trace of the starvation-bound test."""


class PreemptionOverlay:
    def __init__(
        self,
        lock_store: LockStore,
        clearance_s: float = 15.0,
        bus_producer: BusProducer | None = None,
    ) -> None:
        """`clearance_s`: how long before/after a junction's ETA the lock is held --
        the vehicle needs the green starting `clearance_s` before arrival (to clear
        conflicting traffic and complete the yellow/all-red transition) and the lock is
        released `clearance_s` after arrival (once the vehicle has actually passed
        through), per section 4.3's "lock at ETA - (clearance time)".
        """
        self._lock_store = lock_store
        self._clearance_s = clearance_s
        self._bus = bus_producer
        self._runs: dict[str, EmergencyRun] = {}

    def trigger(
        self, run_id: str, route: Sequence[RouteStop], priority: Priority = "emergency"
    ) -> EmergencyRun:
        if not route:
            raise ValueError("route must be non-empty")
        if run_id in self._runs:
            raise ValueError(f"run '{run_id}' already exists")
        run = EmergencyRun(run_id=run_id, route=tuple(route), priority=priority)
        self._runs[run_id] = run
        return run

    def get_run(self, run_id: str) -> EmergencyRun:
        return self._runs[run_id]

    def active_phase_for(self, tls_id: str) -> int | None:
        """The phase a junction under an ACTIVE preemption lock must be forced to
        (docs/design/04_agent_architecture.md section 4.3: "phase to the approach's
        green"), for an IntersectionAgentService (or twin adapter) to consult before
        it lets any other controller (RL/max-pressure/fixed-time) command that
        junction -- enforcement at the consumer, per docs/design/05_communication.md
        section 5.4's "commands with priority=normal are dropped while a lock exists".
        Returns None if no run currently holds a live lock on `tls_id`.
        """
        for run in self._runs.values():
            for stop in run.route:
                if stop.tls_id == tls_id and self._lock_store.holder(tls_id) == run.run_id:
                    return stop.phase_id
        return None

    def advance(self, run_id: str, now_s: float) -> EmergencyRun:
        """Re-evaluates every stop on `run_id`'s route against `now_s`: attempts to
        acquire locks that have entered their window and haven't been tried yet,
        releases locks past their window, and updates `run.state`. Idempotent -- safe
        to call repeatedly with the same or increasing `now_s`.
        """
        run = self._runs[run_id]
        if run.state == "done":
            return run

        for stop in run.route:
            if stop.tls_id in run.locks_released:
                continue

            window_start_s = stop.eta_s - self._clearance_s
            window_end_s = stop.eta_s + self._clearance_s

            if stop.tls_id not in run.locks_acquired:
                if now_s < window_start_s:
                    continue
                if self._lock_store.acquire(
                    stop.tls_id, holder=run_id, ttl_s=2 * self._clearance_s
                ):
                    run.locks_acquired.add(stop.tls_id)
                    run.locks_denied.discard(stop.tls_id)
                    self._publish(run, stop, "lock_acquired")
                else:
                    run.locks_denied.add(stop.tls_id)
                    self._publish(run, stop, "lock_denied")
            elif now_s >= window_end_s:
                self._lock_store.release(stop.tls_id, holder=run_id)
                run.locks_released.add(stop.tls_id)
                self._publish(run, stop, "lock_released")

        run.state = self._derive_state(run, now_s)
        return run

    def _publish(
        self,
        run: EmergencyRun,
        stop: RouteStop,
        event: Literal["lock_acquired", "lock_denied", "lock_released"],
    ) -> None:
        if self._bus is None:
            return
        payload: dict[str, str | int] = {
            "run_id": run.run_id,
            "intersection_id": stop.tls_id,
            "event": event,
            "priority": run.priority,
        }
        if event == "lock_acquired":
            payload["phase_id"] = stop.phase_id
        self._bus.publish("control.preemption.v1", payload)

    def _derive_state(self, run: EmergencyRun, now_s: float) -> RunState:
        route_tls_ids = {stop.tls_id for stop in run.route}
        if run.locks_released == route_tls_ids:
            return "done"
        if run.locks_acquired:
            still_holding = run.locks_acquired - run.locks_released
            return "active" if still_holding else "releasing"
        if now_s >= run.route[0].eta_s - self._clearance_s:
            return "locking"
        return "planned"
