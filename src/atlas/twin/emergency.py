"""Emergency vehicle injection for scenario-driven evaluation (docs/design/
13_roadmap.md Week 7's "emergency scenario class"; docs/design/04_agent_architecture.md
section 4.3). Injects one vehicle of the fleet's own `emergency_vtype` (Week 2's fleet
mix, configs/fleet/manhattan_midtown.yaml) onto an explicit route at a scheduled
sim-time -- the fixture an evaluation script uses to measure the Preemption Overlay's
effect end to end: same twin, same demand, only difference is whether the overlay is
actively locking/forcing green along the vehicle's route.

A separate module from atlas.twin.scenario, not folded into `ScenarioSpec`'s
accident/closure union: those operate on ONE edge over a recurring [start_s,
start_s + duration_s) apply/revert window. An emergency run needs an ORDERED ROUTE
(multiple edges) and departs exactly once -- different enough shape that sharing one
model would force fields onto both that only one of them needs.

`EmergencyVehicleInjector` takes anything satisfying `_TwinLike` (a `.connection` +
`.sim_time` Protocol) structurally, not a concrete `SumoTwinEnv` -- both `SumoTwinEnv`
and `MultiAgentSignalEnv` already expose exactly this pair of properties, so the same
injector works whether the caller is stepping a bare twin or a full multi-agent RL env.
"""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field

from atlas.common.logging import get_logger

logger = get_logger(__name__)


class _TwinLike(Protocol):
    @property
    def connection(self) -> Any: ...
    @property
    def sim_time(self) -> float: ...


class EmergencyVehicleSpec(BaseModel):
    vehicle_id: str
    route_edges: list[str] = Field(min_length=1)
    depart_s: float = Field(ge=0)
    vtype_id: str = "emergency_vtype"
    depart_speed: str = "max"


class EmergencyVehicleInjector:
    """Watches sim_time and injects `spec`'s vehicle exactly once, at `depart_s`."""

    def __init__(self, twin: _TwinLike, spec: EmergencyVehicleSpec) -> None:
        self._twin = twin
        self._spec = spec
        self._injected = False

    def maybe_inject(self) -> None:
        """Call once per sim step (or after every `twin.step()`/`env.step()`) --
        injects the vehicle the first time `sim_time >= depart_s`, then becomes a no-op.
        """
        if self._injected or self._twin.sim_time < self._spec.depart_s:
            return
        conn = self._twin.connection
        route_id = f"{self._spec.vehicle_id}_route"
        conn.route.add(route_id, self._spec.route_edges)
        conn.vehicle.add(
            self._spec.vehicle_id,
            route_id,
            typeID=self._spec.vtype_id,
            departSpeed=self._spec.depart_speed,
        )
        self._injected = True
        logger.info(
            "emergency.vehicle_injected",
            vehicle_id=self._spec.vehicle_id,
            route_edges=self._spec.route_edges,
            sim_time=self._twin.sim_time,
        )

    @property
    def injected(self) -> bool:
        return self._injected

    def is_active(self) -> bool:
        """Whether the vehicle currently exists in the running simulation (departed
        and not yet arrived/removed) -- an evaluation script's signal to keep measuring
        response time, and to know when the vehicle has finished its route.
        """
        if not self._injected:
            return False
        return bool(self._spec.vehicle_id in self._twin.connection.vehicle.getIDList())
