"""LiveTrafficService: the real Manhattan digital twin, running continuously in the
background with ATLAS's own signal controllers in the loop, exposed to the dashboard as a
polled snapshot (per-road speed, per-vehicle position, junction modes, disruptions,
playbook responses and their measured impact).

Why this exists: the dashboard used to draw only incident *status* at 5 junctions -- the
Cognitive API never started a simulation, so there was nothing to draw traffic from, and
approving a playbook changed nothing. This service closes that loop with the same real
network, demand file, controllers and scenario engine the evaluation harness uses (no
synthetic or replayed data):

- A `ControlledTwin` runs in one background thread, one 5-second control decision per
  tick, restarting from t=0 whenever the demand file is exhausted. The 5 corridor
  junctions are driven by ATLAS controllers (fixed-time / actuated / max-pressure,
  switchable per junction) through the safety shield; the other signals keep SUMO's
  native programs.
- Every tick publishes one immutable snapshot dict. Readers (the FastAPI endpoints) just
  grab the latest reference, so they never touch the simulation or block on its lock.
- Submitted incidents (accident / closure / construction / weather) become *real*
  scenario-engine disruptions on the incident's road(s). Incident classes with no road
  effect (sensor_fault, comms_fault, ...) are ignored -- a documented boundary.
- An approved (or auto-applied) playbook is compiled into a real effect in the twin
  (`atlas.twin.playbook_effects`) and its impact is measured: the affected roads' queues
  in the window before vs. after, reported in the snapshot. That is an *observed* change
  in one non-stationary run, not a controlled experiment -- the controlled comparison is
  the what-if tool's job (`atlas.cognitive.what_if`).

Backend choice: the twin runs on `backend="traci"` (a separate SUMO subprocess), NOT
libsumo. libsumo is one global in-process simulation, and the Cognitive API's what-if tool
starts its own libsumo simulation per request -- sharing the binding would have the
what-if runs tear this one down. Every simulation call happens on the one background
thread: the public methods only queue work, they never touch SUMO from a request thread.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import libsumo  # noqa: F401  -- import before sumolib/pyproj, established DLL-order fix
import sumolib

from atlas.common.logging import get_logger
from atlas.twin.controlled_twin import CONTROLLER_MODES, CORRIDOR_TLS_IDS, ControlledTwin
from atlas.twin.incident_effects import (
    ROAD_EFFECT_CLASSES,
    RoadIndex,
    link_ids_of,
    plan_disruption_specs,
)
from atlas.twin.playbook_effects import compile_playbook

logger = get_logger(__name__)

__all__ = ["ROAD_EFFECT_CLASSES", "LiveTrafficService", "plan_disruption_specs"]

DECISION_INTERVAL_S = 5.0


@dataclass(frozen=True, slots=True)
class _PendingIncident:
    incident_id: str
    incident_class: str
    edge_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _PendingPlaybook:
    incident_id: str
    playbook_name: str
    params: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _Disruption:
    name: str
    type: str
    edge_id: str
    incident_id: str
    ends_at_sim_s: float


@dataclass
class _Impact:
    """Before/after queue measurement around one playbook application."""

    incident_id: str
    playbook: str
    summary: str
    edge_ids: tuple[str, ...]
    applied_at_s: float
    window_s: float
    before_halted: float | None
    before_speed_ratio: float | None
    after_halted: list[float] = field(default_factory=list)
    after_speed_ratio: list[float] = field(default_factory=list)
    status: str = "measuring"  # measuring | complete | no_change | not_applied

    def as_dict(self, now_s: float) -> dict[str, object]:
        after_halted = _mean(self.after_halted)
        after_ratio = _mean(self.after_speed_ratio)
        change_pct: float | None = None
        if self.before_halted and after_halted is not None and self.before_halted > 0:
            change_pct = round((after_halted - self.before_halted) / self.before_halted * 100, 1)
        return {
            "incident_id": self.incident_id,
            "playbook": self.playbook,
            "summary": self.summary,
            "status": self.status,
            "edge_ids": list(self.edge_ids),
            "window_s": self.window_s,
            "elapsed_s": max(0.0, now_s - self.applied_at_s),
            "before_halted": _round(self.before_halted),
            "after_halted": _round(after_halted),
            "before_speed_ratio": _round(self.before_speed_ratio, 3),
            "after_speed_ratio": _round(after_ratio, 3),
            "halted_change_pct": change_pct,
        }


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _round(value: float | None, digits: int = 1) -> float | None:
    return None if value is None else round(value, digits)


class LiveTrafficService:
    def __init__(
        self,
        net_file: Path,
        route_file: Path,
        additional_files: Sequence[Path] = (),
        *,
        decisions_per_tick: int = 1,
        tick_interval_s: float = 1.0,
        warmup_s: int = 300,
        disruption_duration_s: int = 900,
        impact_window_s: float = 300.0,
        seed: int = 42,
        tls_ids: Sequence[str] = CORRIDOR_TLS_IDS,
    ) -> None:
        self._decisions_per_tick = decisions_per_tick
        self._tick_interval_s = tick_interval_s
        self._warmup_decisions = max(1, round(warmup_s / DECISION_INTERVAL_S))
        self._disruption_duration_s = disruption_duration_s
        self._impact_window_s = impact_window_s
        self._seed = seed

        # Read the static network once: geometry is served from it even before (or
        # without) a running simulation.
        self._net: Any = sumolib.net.readNet(str(net_file))
        self._index = RoadIndex.from_net(self._net)
        self._edges = [e for e in self._net.getEdges() if not e.getID().startswith(":")]
        self._edge_ids = [e.getID() for e in self._edges]
        self._geometry = self._build_geometry()

        self._twin = ControlledTwin(
            net_file,
            route_file,
            additional_files,
            tls_ids=tls_ids,
            backend="traci",
            decision_interval_s=DECISION_INTERVAL_S,
            road_index=self._index,
        )
        self._tls_ids = list(tls_ids)
        self._modes: dict[str, str] = {}  # operator-chosen base modes; survive restarts
        self._cycle = 0
        self._disruptions: list[_Disruption] = []
        self._impacts: dict[str, _Impact] = {}
        # (sim_time, halted per edge, speed ratio per edge) per tick, for the "before" window
        self._history: deque[tuple[float, dict[str, int], dict[str, float | None]]] = deque(
            maxlen=max(10, round(impact_window_s / (DECISION_INTERVAL_S * decisions_per_tick)) + 5)
        )
        self._pending_incidents: list[_PendingIncident] = []
        self._pending_playbooks: list[_PendingPlaybook] = []
        self._pending_lock = threading.Lock()
        self._known_incident_ids: set[str] = set()
        self._known_playbook_ids: set[str] = set()
        self._snapshot: dict[str, object] = self._empty_snapshot(ready=False, error=None)

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._ready = threading.Event()

    # -- public API (safe to call from any thread) --

    def geometry(self) -> list[dict[str, object]]:
        return self._geometry

    def snapshot(self) -> dict[str, object]:
        return self._snapshot

    def apply_incident(self, incident: Mapping[str, object]) -> list[str]:
        """Queues a real disruption for the incident's road(s) and returns the edge ids
        it will affect (empty if the incident class has no road effect or its location
        doesn't resolve to any road in the network). Applied by the simulation thread on
        its next tick.

        `location.link_ids` may hold edge ids (that road) or junction ids (every road
        entering that junction) -- the dashboard's map keys incidents by junction.
        """
        incident_class = str(incident.get("class", ""))
        if incident_class not in ROAD_EFFECT_CLASSES:
            return []
        incident_id = str(incident.get("incident_id", ""))
        if not incident_id:
            return []

        edge_ids = self._index.resolve(link_ids_of(incident.get("location")))
        if not edge_ids:
            logger.warning("live_traffic.incident_unresolved", incident_id=incident_id)
            return []

        with self._pending_lock:
            if incident_id in self._known_incident_ids:
                return []
            self._known_incident_ids.add(incident_id)
            self._pending_incidents.append(
                _PendingIncident(incident_id, incident_class, tuple(edge_ids))
            )
        return edge_ids

    def apply_playbook(
        self, incident_id: str, playbook_name: str, params: Mapping[str, object]
    ) -> bool:
        """Queues a playbook response for the simulation thread to compile and apply.
        Returns False if this incident already had one queued (a response is applied at
        most once per incident)."""
        with self._pending_lock:
            if incident_id in self._known_playbook_ids:
                return False
            self._known_playbook_ids.add(incident_id)
            self._pending_playbooks.append(_PendingPlaybook(incident_id, playbook_name, params))
        return True

    def set_junction_mode(self, tls_id: str, mode: str) -> None:
        """Sets an ATLAS-controlled junction's controller mode (applied on the next
        decision). Raises KeyError for a junction ATLAS doesn't control and ValueError
        for a mode that isn't available live (`rl` has no trained policy)."""
        if tls_id not in self._tls_ids:
            raise KeyError(f"'{tls_id}' is not an ATLAS-controlled junction")
        if mode not in CONTROLLER_MODES:
            raise ValueError(f"mode must be one of {list(CONTROLLER_MODES)}, got '{mode}'")
        self._modes[tls_id] = mode
        if self._twin.is_running:
            self._twin.set_mode(tls_id, mode)

    def start(self) -> None:
        """Starts the background simulation thread (idempotent)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="live-traffic", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
        self._close_simulation()

    def wait_until_ready(self, timeout_s: float) -> bool:
        return self._ready.wait(timeout_s)

    # -- simulation lifecycle (single-threaded: only the background thread, or a test
    #    calling these directly in place of it, may call them) --

    def start_simulation(self) -> None:
        """Starts a fresh simulation and runs the warm-up so the first published
        snapshot already has traffic on the roads rather than an empty network."""
        self._close_simulation()
        self._twin.reset(seed=self._seed + self._cycle)
        self._cycle += 1
        for tls_id, mode in self._modes.items():
            self._twin.set_mode(tls_id, mode)
        self._disruptions = []
        self._impacts = {}
        self._history.clear()
        self._twin.step_decisions(self._warmup_decisions)
        self._snapshot = self._build_snapshot()
        self._ready.set()
        logger.info("live_traffic.started", cycle=self._cycle)

    def tick(self) -> None:
        """Advances the simulation one tick and publishes a new snapshot."""
        if not self._twin.is_running:
            raise RuntimeError("tick() called before start_simulation()")
        self._apply_pending_incidents()
        self._apply_pending_playbooks()
        self._twin.step_decisions(self._decisions_per_tick)
        sim_time = self._twin.sim_time
        self._disruptions = [d for d in self._disruptions if d.ends_at_sim_s > sim_time]
        self._snapshot = self._build_snapshot()

    def simulation_finished(self) -> bool:
        """True once every scheduled vehicle has arrived (the demand file is spent)."""
        return self._twin.is_running and self._twin.finished()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                if not self._twin.is_running:
                    self.start_simulation()
                started = time.monotonic()
                self.tick()
                if self.simulation_finished():
                    logger.info("live_traffic.cycle_complete", cycle=self._cycle)
                    self._close_simulation()
                    continue
                self._stop.wait(max(0.0, self._tick_interval_s - (time.monotonic() - started)))
            except Exception as exc:  # the live map must never take the API down with it
                logger.exception("live_traffic.error")
                self._snapshot = self._empty_snapshot(ready=False, error=str(exc))
                self._close_simulation()
                self._stop.wait(5.0)

    def _close_simulation(self) -> None:
        try:
            self._twin.close()
        except Exception:
            logger.exception("live_traffic.close_error")

    # -- internals --

    def _apply_pending_incidents(self) -> None:
        with self._pending_lock:
            pending, self._pending_incidents = self._pending_incidents, []
        if not pending:
            return
        sim_time = self._twin.sim_time
        for item in pending:
            specs = plan_disruption_specs(
                item.incident_id,
                item.incident_class,
                {edge_id: self._index.lane_counts[edge_id] for edge_id in item.edge_ids},
                start_s=int(sim_time),
                duration_s=self._disruption_duration_s,
            )
            self._twin.add_disruptions(specs)
            for spec in specs:
                self._disruptions.append(
                    _Disruption(
                        name=spec.name,
                        type=spec.type,
                        edge_id=spec.edge_id,
                        incident_id=item.incident_id,
                        ends_at_sim_s=sim_time + self._disruption_duration_s,
                    )
                )
            logger.info(
                "live_traffic.disruption_queued",
                incident_id=item.incident_id,
                incident_class=item.incident_class,
                edges=list(item.edge_ids),
            )

    def _apply_pending_playbooks(self) -> None:
        with self._pending_lock:
            pending, self._pending_playbooks = self._pending_playbooks, []
        if not pending:
            return
        sim_time = self._twin.sim_time
        for item in pending:
            effect = compile_playbook(
                item.playbook_name,
                item.params,
                self._index,
                self._tls_ids,
                default_duration_s=float(self._disruption_duration_s),
            )
            if effect is None:
                self._impacts[item.incident_id] = _Impact(
                    item.incident_id,
                    item.playbook_name,
                    "This playbook cannot be applied to the live twin",
                    (),
                    sim_time,
                    self._impact_window_s,
                    None,
                    None,
                    status="not_applied",
                )
                continue
            if effect.is_noop:
                self._impacts[item.incident_id] = _Impact(
                    item.incident_id,
                    item.playbook_name,
                    effect.summary,
                    (),
                    sim_time,
                    self._impact_window_s,
                    None,
                    None,
                    status="no_change",
                )
                continue

            before_halted, before_ratio = self._before_window(effect.affected_edges, sim_time)
            self._twin.apply_effect(item.incident_id, effect)
            self._impacts[item.incident_id] = _Impact(
                item.incident_id,
                item.playbook_name,
                effect.summary,
                effect.affected_edges,
                sim_time,
                self._impact_window_s,
                before_halted,
                before_ratio,
            )

    def _before_window(
        self, edge_ids: Sequence[str], now_s: float
    ) -> tuple[float | None, float | None]:
        """Mean queued vehicles / speed ratio on `edge_ids` over the window before now."""
        halted: list[float] = []
        ratios: list[float] = []
        for t, halted_by_edge, ratio_by_edge in self._history:
            if t <= now_s - self._impact_window_s:
                continue
            halted.append(float(sum(halted_by_edge.get(e, 0) for e in edge_ids)))
            vals = [r for e in edge_ids if (r := ratio_by_edge.get(e)) is not None]
            if vals:
                ratios.append(sum(vals) / len(vals))
        return _mean(halted), _mean(ratios)

    def _record_impact_samples(
        self, sim_time: float, halted: dict[str, int], ratios: dict[str, float | None]
    ) -> None:
        for impact in self._impacts.values():
            if impact.status != "measuring" or sim_time <= impact.applied_at_s:
                continue
            impact.after_halted.append(float(sum(halted.get(e, 0) for e in impact.edge_ids)))
            vals = [r for e in impact.edge_ids if (r := ratios.get(e)) is not None]
            if vals:
                impact.after_speed_ratio.append(sum(vals) / len(vals))
            if sim_time >= impact.applied_at_s + impact.window_s:
                impact.status = "complete"

    def _build_geometry(self) -> list[dict[str, object]]:
        geometry: list[dict[str, object]] = []
        for edge in self._edges:
            coords = [
                [round(lon, 6), round(lat, 6)]
                for lon, lat in (self._net.convertXY2LonLat(x, y) for x, y in edge.getShape())
            ]
            geometry.append(
                {
                    "edge_id": edge.getID(),
                    "lanes": self._index.lane_counts[edge.getID()],
                    "speed_limit_mps": round(self._index.free_flow_mps[edge.getID()], 2),
                    "coords": coords,
                }
            )
        return geometry

    def _empty_snapshot(self, ready: bool, error: str | None) -> dict[str, object]:
        return {
            "ready": ready,
            "error": error,
            "sim_time_s": 0.0,
            "stats": {"vehicles": 0, "halted": 0, "mean_speed_kmh": 0.0},
            "edges": {},
            "vehicles": [],
            "disruptions": [],
            "junctions": {},
            "playbooks": [],
            "impacts": {},
        }

    def _build_snapshot(self) -> dict[str, object]:
        conn = self._twin.connection
        sim_time = float(self._twin.sim_time)

        edges: dict[str, dict[str, object]] = {}
        halted_by_edge: dict[str, int] = {}
        ratio_by_edge: dict[str, float | None] = {}
        for edge_id in self._edge_ids:
            count = int(conn.edge.getLastStepVehicleNumber(edge_id))
            ratio: float | None = None
            if count > 0:
                free_flow = self._index.free_flow_mps[edge_id]
                speed = float(conn.edge.getLastStepMeanSpeed(edge_id))
                ratio = round(min(1.0, speed / free_flow), 3) if free_flow > 0 else None
            edges[edge_id] = {"vehicles": count, "speed_ratio": ratio}
            halted_by_edge[edge_id] = int(conn.edge.getLastStepHaltingNumber(edge_id))
            ratio_by_edge[edge_id] = ratio

        self._history.append((sim_time, halted_by_edge, ratio_by_edge))
        self._record_impact_samples(sim_time, halted_by_edge, ratio_by_edge)

        vehicles: list[list[float]] = []
        speeds: list[float] = []
        for vehicle_id in conn.vehicle.getIDList():
            x, y = conn.vehicle.getPosition(vehicle_id)
            lon, lat = self._net.convertXY2LonLat(x, y)
            speed = float(conn.vehicle.getSpeed(vehicle_id))
            speeds.append(speed)
            vehicles.append([round(lon, 6), round(lat, 6), round(speed, 1)])

        mean_speed_kmh = round(sum(speeds) / len(speeds) * 3.6, 1) if speeds else 0.0
        return {
            "ready": True,
            "error": None,
            "sim_time_s": sim_time,
            "stats": {
                "vehicles": len(vehicles),
                "halted": sum(1 for s in speeds if s < 0.1),
                "mean_speed_kmh": mean_speed_kmh,
            },
            "edges": edges,
            "vehicles": vehicles,
            "disruptions": [
                {
                    "name": d.name,
                    "type": d.type,
                    "edge_id": d.edge_id,
                    "incident_id": d.incident_id,
                    "ends_at_sim_s": d.ends_at_sim_s,
                }
                for d in self._disruptions
            ],
            "junctions": self._twin.junction_states(),
            "playbooks": [
                {
                    "incident_id": a.effect_id,
                    "playbook": a.effect.playbook_name,
                    "summary": a.effect.summary,
                    "ends_at_sim_s": a.ends_at_s,
                }
                for a in self._twin.active_effects()
            ],
            "impacts": {i: imp.as_dict(sim_time) for i, imp in self._impacts.items()},
        }
