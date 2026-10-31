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
from typing import Any, Literal

import libsumo  # noqa: F401  -- import before sumolib/pyproj, established DLL-order fix
import sumolib
from traci import constants as tc

from atlas.common.logging import get_logger
from atlas.control.signal_index import NetworkSignalIndex
from atlas.marl.policy_io import LoadedPolicy
from atlas.twin.controlled_twin import DISTRICT_TLS_IDS, ControlledTwin
from atlas.twin.env import TwinBackend
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
# Replay buffer: ~15 minutes of 1 Hz frames. In-memory on purpose (a few MB): it feeds the
# dashboard's replay scrubber for "what just happened", not long-term telemetry (the design's
# Timescale store, ADR-0003, is the home for that and is not built).
HISTORY_FRAMES = 900
# Above this many roads the snapshot goes compact: only roads that currently carry traffic are
# listed, geometry coordinates are rounded to ~1 m, and the replay buffer is shorter -- a whole
# borough (11k roads, 2.7k signals) would otherwise cost megabytes per snapshot.
COMPACT_EDGE_THRESHOLD = 2000
COMPACT_HISTORY_FRAMES = 100
DEFAULT_MAX_VEHICLES = 2500
_EDGE_VARS = [
    tc.LAST_STEP_VEHICLE_NUMBER,
    tc.LAST_STEP_MEAN_SPEED,
    tc.LAST_STEP_VEHICLE_HALTING_NUMBER,
]
_VEHICLE_VARS = [tc.VAR_POSITION, tc.VAR_SPEED]


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
        tls_ids: Sequence[str] | Literal["all"] = DISTRICT_TLS_IDS,
        policy: LoadedPolicy | None = None,
        backend: TwinBackend = "traci",
        time_to_teleport_s: float = -1,
        scale: float = 1.0,
        max_vehicles: int = DEFAULT_MAX_VEHICLES,
        history_frames: int | None = None,
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
        self._compact = len(self._edge_ids) > COMPACT_EDGE_THRESHOLD
        self._max_vehicles = max_vehicles
        self._geometry = self._build_geometry()

        # One streaming read of the network for every signal's control structure (not one parse per
        # junction); also what lets tls_ids="all" mean "every signal ATLAS can control".
        signals = NetworkSignalIndex.from_net_file(net_file, None if tls_ids == "all" else tls_ids)
        self._twin = ControlledTwin(
            net_file,
            route_file,
            additional_files,
            tls_ids=tls_ids,
            backend=backend,
            decision_interval_s=DECISION_INTERVAL_S,
            road_index=self._index,
            policy=policy,
            signal_index=signals,
            time_to_teleport_s=time_to_teleport_s,
            scale=scale,
        )
        self._tls_ids = self._twin.tls_ids
        self._subscribed_vehicles: set[str] = set()
        self._x_real_time: float | None = None
        self._junction_positions = self._build_junction_positions()
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
        self._frames: deque[dict[str, object]] = deque(
            maxlen=history_frames or (COMPACT_HISTORY_FRAMES if self._compact else HISTORY_FRAMES)
        )
        self._frames_lock = threading.Lock()

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._ready = threading.Event()

    # -- public API (safe to call from any thread) --

    def geometry(self) -> list[dict[str, object]]:
        return self._geometry

    def junction_positions(self) -> list[dict[str, object]]:
        """Where each ATLAS-controlled junction is (lon/lat), for the map's markers."""
        return self._junction_positions

    def snapshot(self) -> dict[str, object]:
        return self._snapshot

    def history(self, max_frames: int = 300, stride: int = 1) -> list[dict[str, object]]:
        """Recent frames, oldest first, ending with the latest: every `stride`-th frame
        counting back from now, at most `max_frames` of them. Frames are the same dicts
        `snapshot()` published (never mutated afterwards), so replaying one in the
        dashboard shows exactly what was live then."""
        if max_frames < 1 or stride < 1:
            raise ValueError("max_frames and stride must be >= 1")
        with self._frames_lock:
            frames = list(self._frames)
        picked = frames[::-1][::stride][:max_frames]
        return picked[::-1]

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
        if mode not in self._twin.available_modes:
            raise ValueError(
                f"mode must be one of {list(self._twin.available_modes)}, got '{mode}'"
            )
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
        with self._frames_lock:
            self._frames.clear()  # a new run restarts sim time; old frames would interleave
        self._subscribe_edges()
        self._twin.step_decisions(self._warmup_decisions)
        self._publish(self._build_snapshot())
        self._ready.set()
        logger.info("live_traffic.started", cycle=self._cycle)

    def tick(self) -> None:
        """Advances the simulation one tick and publishes a new snapshot."""
        if not self._twin.is_running:
            raise RuntimeError("tick() called before start_simulation()")
        self._apply_pending_incidents()
        self._apply_pending_playbooks()
        sim_before, wall_before = self._twin.sim_time, time.monotonic()
        self._twin.step_decisions(self._decisions_per_tick)
        sim_time = self._twin.sim_time
        wall = time.monotonic() - wall_before
        if wall > 0:
            speed = (sim_time - sim_before) / wall
            self._x_real_time = (
                speed if self._x_real_time is None else 0.8 * self._x_real_time + 0.2 * speed
            )
        self._disruptions = [d for d in self._disruptions if d.ends_at_sim_s > sim_time]
        self._publish(self._build_snapshot())

    def _publish(self, snapshot: dict[str, object]) -> None:
        self._snapshot = snapshot
        with self._frames_lock:
            self._frames.append(snapshot)

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

    def _build_junction_positions(self) -> list[dict[str, object]]:
        """Lon/lat per controlled junction. A signal id is normally a node id; SUMO's grouped
        signals (`GS_...`) are not, so those sit at the centroid of the nodes the signal
        actually controls."""
        positions: list[dict[str, object]] = []
        for tls_id in self._tls_ids:
            if self._net.hasNode(tls_id):
                x, y = self._net.getNode(tls_id).getCoord()
            else:
                nodes = {
                    conn[0].getEdge().getToNode().getID(): conn[0].getEdge().getToNode()
                    for conn in self._net.getTLS(tls_id).getConnections()
                }
                coords = [n.getCoord() for n in nodes.values()]
                x = sum(c[0] for c in coords) / len(coords)
                y = sum(c[1] for c in coords) / len(coords)
            lon, lat = self._net.convertXY2LonLat(x, y)
            positions.append({"id": tls_id, "lon": round(lon, 6), "lat": round(lat, 6)})
        return positions

    def _build_geometry(self) -> list[dict[str, object]]:
        geometry: list[dict[str, object]] = []
        digits = 5 if self._compact else 6
        for edge in self._edges:
            coords = [
                [round(lon, digits), round(lat, digits)]
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

    def _subscribe_edges(self) -> None:
        """One subscription per road (for the life of this simulation) so each tick reads every
        road's count/speed/queue in a single call instead of three calls per road."""
        conn = self._twin.connection
        for edge_id in self._edge_ids:
            conn.edge.subscribe(edge_id, _EDGE_VARS)
        self._subscribed_vehicles = set()

    def _read_vehicles(self, conn: Any) -> tuple[list[list[float]], list[float]]:
        """Positions for at most `max_vehicles` evenly sampled vehicles, and speeds for ALL of
        them (the stats stay exact; only the drawn dots are sampled). Vehicles are subscribed the
        first tick they are seen, so later ticks cost one call, not one per vehicle."""
        ids = list(conn.vehicle.getIDList())
        current = set(ids)
        for vehicle_id in current - self._subscribed_vehicles:
            conn.vehicle.subscribe(vehicle_id, _VEHICLE_VARS)
        self._subscribed_vehicles = current
        results = conn.vehicle.getAllSubscriptionResults()

        stride = max(1, -(-len(ids) // self._max_vehicles)) if self._max_vehicles else 1
        speeds: list[float] = []
        vehicles: list[list[float]] = []
        for index, vehicle_id in enumerate(ids):
            values = results.get(vehicle_id)
            if values:
                x, y = values[tc.VAR_POSITION]
                speed = float(values[tc.VAR_SPEED])
            else:  # subscribed just now: its first results arrive after the next step
                x, y = conn.vehicle.getPosition(vehicle_id)
                speed = float(conn.vehicle.getSpeed(vehicle_id))
            speeds.append(speed)
            if index % stride == 0:
                lon, lat = self._net.convertXY2LonLat(x, y)
                vehicles.append([round(lon, 6), round(lat, 6), round(speed, 1)])
        return vehicles, speeds

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
            "available_modes": [],
            "playbooks": [],
            "impacts": {},
        }

    def _build_snapshot(self) -> dict[str, object]:
        conn = self._twin.connection
        sim_time = float(self._twin.sim_time)

        edges: dict[str, dict[str, object]] = {}
        halted_by_edge: dict[str, int] = {}
        ratio_by_edge: dict[str, float | None] = {}
        results = conn.edge.getAllSubscriptionResults()
        for edge_id in self._edge_ids:
            values = results.get(edge_id)
            count = int(values[tc.LAST_STEP_VEHICLE_NUMBER]) if values else 0
            ratio: float | None = None
            if values and count > 0:
                free_flow = self._index.free_flow_mps[edge_id]
                speed = float(values[tc.LAST_STEP_MEAN_SPEED])
                ratio = round(min(1.0, speed / free_flow), 3) if free_flow > 0 else None
            if count > 0 or not self._compact:
                edges[edge_id] = {"vehicles": count, "speed_ratio": ratio}
            # Sparse on purpose: consumers use .get(edge, 0/None), and dense per-tick dicts of every
            # road are what blow up the history window at city scale.
            halted = int(values[tc.LAST_STEP_VEHICLE_HALTING_NUMBER]) if values else 0
            if halted:
                halted_by_edge[edge_id] = halted
            if ratio is not None:
                ratio_by_edge[edge_id] = ratio

        self._history.append((sim_time, halted_by_edge, ratio_by_edge))
        self._record_impact_samples(sim_time, halted_by_edge, ratio_by_edge)

        vehicles, speeds = self._read_vehicles(conn)

        mean_speed_kmh = round(sum(speeds) / len(speeds) * 3.6, 1) if speeds else 0.0
        return {
            "ready": True,
            "error": None,
            "sim_time_s": sim_time,
            "stats": {
                "vehicles": len(speeds),
                "halted": sum(1 for s in speeds if s < 0.1),
                "mean_speed_kmh": mean_speed_kmh,
                "x_real_time": round(self._x_real_time, 1) if self._x_real_time else None,
                "shown_vehicles": len(vehicles),
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
            "available_modes": list(self._twin.available_modes),
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
