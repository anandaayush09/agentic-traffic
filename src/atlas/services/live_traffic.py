"""LiveTrafficService: the real Manhattan digital twin, running continuously in the
background, exposed to the dashboard as a polled snapshot (per-road speed, per-vehicle
position, active disruptions).

Why this exists: the Week 10 dashboard only ever drew incident *status* at 5 junctions --
the Cognitive API never started a simulation, so there was nothing to draw traffic from.
This service closes that gap with the same real network, demand file and scenario engine
the evaluation harness uses (no synthetic or replayed data):

- A `SumoTwinEnv` runs in one background thread, a few simulated seconds per wall-clock
  tick, restarting from t=0 whenever the demand file is exhausted.
- Every tick publishes one immutable snapshot dict. Readers (the FastAPI endpoints) just
  grab the latest reference, so they never touch the simulation or block on its lock.
- Submitted incidents (accident / closure / construction / weather) become *real*
  scenario-engine disruptions on the incident's road(s): an accident slows the lanes, a
  closure blocks them, and traffic visibly backs up and reroutes. Incident classes with
  no road effect (sensor_fault, comms_fault, ...) are ignored here -- a documented
  boundary, not silently treated as covered.

Backend choice: the twin runs on `backend="traci"` (a separate SUMO subprocess), NOT
libsumo. libsumo is one global in-process simulation, and the Cognitive API's what-if tool
already starts its own libsumo simulation per request -- sharing the binding would have
the what-if runs tear this one down. traci is a different module with its own socket
connection, so the two coexist. Every simulation call happens on the one background
thread: `apply_incident` only queues work, it never touches SUMO from a request thread.

Approving an incident does not (yet) apply its playbook to this twin -- the supervisor's
what-if tool and this live twin are separate simulations, and a playbook-to-simulation
effect compiler doesn't exist (see what_if.py). The disruption simply runs for
`disruption_duration_s` simulated seconds, then clears.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import libsumo  # noqa: F401  -- import before sumolib/pyproj, established DLL-order fix
import sumolib

from atlas.common.logging import get_logger
from atlas.twin import ScenarioController, ScenarioSpec, SumoTwinEnv, TwinEnvConfig

logger = get_logger(__name__)

# Incident classes that have a physical effect on a road. Everything else (sensor_fault,
# comms_fault, event_surge, emergency_run, ...) is a no-op for the live map.
ROAD_EFFECT_CLASSES = frozenset({"accident", "closure", "construction", "weather"})

_ACCIDENT_SPEED_FACTOR = 0.25
_WEATHER_SPEED_FACTOR = 0.6
_CONSTRUCTION_SPEED_FACTOR = 0.7
_SINGLE_LANE_CONSTRUCTION_SPEED_FACTOR = 0.5


def plan_disruption_specs(
    incident_id: str,
    incident_class: str,
    edge_lane_counts: Mapping[str, int],
    start_s: int,
    duration_s: int,
) -> list[ScenarioSpec]:
    """The pure mapping from one incident to scenario-engine specs, one per affected
    edge (so each can be tracked and reverted independently). Same severities the
    committed scenario config (`configs/scenarios/manhattan_midtown.yaml`) uses for
    its first variant of each class.
    """
    specs: list[ScenarioSpec] = []
    for edge_id, lane_count in edge_lane_counts.items():
        name = f"{incident_id}:{edge_id}"
        common: dict[str, Any] = {
            "name": name,
            "edge_id": edge_id,
            "start_s": start_s,
            "duration_s": duration_s,
        }
        if incident_class == "accident":
            specs.append(
                ScenarioSpec(type="accident", speed_factor=_ACCIDENT_SPEED_FACTOR, **common)
            )
        elif incident_class == "closure":
            specs.append(ScenarioSpec(type="closure", **common))
        elif incident_class == "weather":
            specs.append(ScenarioSpec(type="weather", speed_factor=_WEATHER_SPEED_FACTOR, **common))
        elif incident_class == "construction":
            if lane_count >= 2:
                # A real lane drop: block the first lane, derate the rest.
                specs.append(
                    ScenarioSpec(
                        type="construction",
                        speed_factor=_CONSTRUCTION_SPEED_FACTOR,
                        lane_indices=[0],
                        **common,
                    )
                )
            else:
                # A single-lane edge has no lane to drop without closing it outright.
                specs.append(
                    ScenarioSpec(
                        type="accident",
                        speed_factor=_SINGLE_LANE_CONSTRUCTION_SPEED_FACTOR,
                        **common,
                    )
                )
    return specs


@dataclass(frozen=True, slots=True)
class _PendingIncident:
    incident_id: str
    incident_class: str
    edge_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Disruption:
    name: str
    type: str
    edge_id: str
    incident_id: str
    ends_at_sim_s: float


class LiveTrafficService:
    def __init__(
        self,
        net_file: Path,
        route_file: Path,
        additional_files: Sequence[Path] = (),
        *,
        sim_seconds_per_tick: int = 4,
        tick_interval_s: float = 1.0,
        warmup_s: int = 300,
        disruption_duration_s: int = 900,
        seed: int = 42,
    ) -> None:
        self._net_file = net_file
        self._route_file = route_file
        self._additional_files = list(additional_files)
        self._sim_seconds_per_tick = sim_seconds_per_tick
        self._tick_interval_s = tick_interval_s
        self._warmup_s = warmup_s
        self._disruption_duration_s = disruption_duration_s
        self._seed = seed

        # Read the static network once: geometry is served from it even before (or
        # without) a running simulation.
        self._net: Any = sumolib.net.readNet(str(net_file))
        self._edges = [e for e in self._net.getEdges() if not e.getID().startswith(":")]
        self._edge_ids = [e.getID() for e in self._edges]
        self._free_flow_mps = {e.getID(): float(e.getSpeed()) for e in self._edges}
        self._lane_counts = {e.getID(): int(e.getLaneNumber()) for e in self._edges}
        self._geometry = self._build_geometry()

        self._env: SumoTwinEnv | None = None
        self._controller: ScenarioController | None = None
        self._cycle = 0
        self._disruptions: list[_Disruption] = []
        self._pending: list[_PendingIncident] = []
        self._pending_lock = threading.Lock()
        self._known_incident_ids: set[str] = set()
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

        edge_ids = self._resolve_edges(incident.get("location"))
        if not edge_ids:
            logger.warning("live_traffic.incident_unresolved", incident_id=incident_id)
            return []

        with self._pending_lock:
            if incident_id in self._known_incident_ids:
                return []
            self._known_incident_ids.add(incident_id)
            self._pending.append(_PendingIncident(incident_id, incident_class, tuple(edge_ids)))
        return edge_ids

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
        config = TwinEnvConfig(
            name="live_traffic",
            net_file=self._net_file,
            route_file=self._route_file,
            additional_files=self._additional_files,
            backend="traci",
            # A closure invalidates routes that run through it; without rerouting those
            # vehicles would just be dropped instead of finding another way round.
            enable_rerouting=True,
        )
        env = SumoTwinEnv(config)
        env.reset(seed=self._seed + self._cycle)
        self._cycle += 1
        self._env = env
        self._controller = ScenarioController(env, [])
        self._disruptions = []
        env.step(self._warmup_s)
        self._snapshot = self._build_snapshot()
        self._ready.set()
        logger.info("live_traffic.started", cycle=self._cycle, warmup_s=self._warmup_s)

    def tick(self) -> None:
        """Advances the simulation one tick and publishes a new snapshot."""
        controller = self._controller
        if controller is None or self._env is None:
            raise RuntimeError("tick() called before start_simulation()")
        self._apply_pending(controller)
        controller.step(self._sim_seconds_per_tick)
        sim_time = self._env.sim_time
        self._disruptions = [d for d in self._disruptions if d.ends_at_sim_s > sim_time]
        self._snapshot = self._build_snapshot()

    def simulation_finished(self) -> bool:
        """True once every scheduled vehicle has arrived (the demand file is spent)."""
        if self._env is None:
            return False
        return int(self._env.connection.simulation.getMinExpectedNumber()) == 0

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                if self._env is None:
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
        env, self._env, self._controller = self._env, None, None
        if env is not None:
            try:
                env.close()
            except Exception:
                logger.exception("live_traffic.close_error")

    # -- internals --

    def _resolve_edges(self, location: object) -> list[str]:
        link_ids: list[str] = []
        if isinstance(location, Mapping):
            raw = location.get("link_ids")
            if isinstance(raw, list):
                link_ids = [str(x) for x in raw]
        resolved: dict[str, None] = {}
        for link_id in link_ids:
            if link_id in self._lane_counts:
                resolved.setdefault(link_id, None)
            elif self._net.hasNode(link_id):
                for edge in self._net.getNode(link_id).getIncoming():
                    if edge.getID() in self._lane_counts:
                        resolved.setdefault(edge.getID(), None)
        return list(resolved)

    def _apply_pending(self, controller: ScenarioController) -> None:
        with self._pending_lock:
            pending, self._pending = self._pending, []
        if not pending or self._env is None:
            return
        sim_time = self._env.sim_time
        for item in pending:
            specs = plan_disruption_specs(
                item.incident_id,
                item.incident_class,
                {edge_id: self._lane_counts[edge_id] for edge_id in item.edge_ids},
                start_s=int(sim_time),
                duration_s=self._disruption_duration_s,
            )
            for spec in specs:
                controller.add_spec(spec)
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
                    "lanes": self._lane_counts[edge.getID()],
                    "speed_limit_mps": round(self._free_flow_mps[edge.getID()], 2),
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
        }

    def _build_snapshot(self) -> dict[str, object]:
        assert self._env is not None
        conn = self._env.connection

        edges: dict[str, dict[str, object]] = {}
        for edge_id in self._edge_ids:
            count = int(conn.edge.getLastStepVehicleNumber(edge_id))
            ratio: float | None = None
            if count > 0:
                free_flow = self._free_flow_mps[edge_id]
                speed = float(conn.edge.getLastStepMeanSpeed(edge_id))
                ratio = round(min(1.0, speed / free_flow), 3) if free_flow > 0 else None
            edges[edge_id] = {"vehicles": count, "speed_ratio": ratio}

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
            "sim_time_s": float(self._env.sim_time),
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
        }
