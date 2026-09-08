"""Scenario engine (docs/design/08_data_strategy.md section 8.3's 10 disruption
classes) applied to a running digital twin. Rush-hour surge is already covered by the
am_peak/pm_peak demand periods (atlas.twin.demand), and emergency run/VIP movement by
`atlas.twin.emergency.EmergencyVehicleInjector` (Week 7) -- neither is a
`ScenarioSpec` type here. Sensor failure and comms failure are separate fault-model
concerns (`atlas.twin.sensor_fault`, and bus-partition chaos tests respectively), not
edge/lane effects a twin scenario applies. Public event (localized OD spike) is
demand-generation work, not a twin-state effect, and is not built yet -- a real,
named gap, not silently assumed covered by one of the types below.

Mechanism (verified against libsumo directly before writing this):
- **closure**: `lane.setAllowed(lane_id, [])` on every lane of the target edge -- sets
  the allowed-vehicle-class list to empty, blocking all classes. NOT the same as clearing
  a restriction; reverting requires restoring the lane's original `getAllowed()` tuple.
- **accident**: `lane.setMaxSpeed(lane_id, original_speed * speed_factor)` on every lane
  of the target edge -- a slowdown rather than a full block, distinguishing it from a
  closure. Reverting restores the original max speed.
- **construction** (Week 11): "long-duration lane drop + speed limit" made literal --
  `lane_indices` selects a SUBSET of the edge's lanes to fully block (closure's
  mechanism, scoped to those lanes only), while `speed_factor` derates the edge's
  REMAINING (unblocked) lanes -- a real lane drop plus a real speed limit on what's
  left, not just accident/closure applied edge-wide.
- **weather** (Week 11): the same speed-derate mechanism as accident, but applied
  across `edge_id` plus every edge in `area_edge_ids` at once -- "global/area
  speed... derate" made literal as "more than one edge, same factor," not a new
  mechanism.

`ScenarioController` wraps `SumoTwinEnv.step()` and applies/reverts scenarios by watching
`sim_time` cross each spec's [start_s, start_s + duration_s) window, so callers just call
`controller.step()` in a loop exactly as they would call `env.step()` directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from atlas.common.logging import get_logger
from atlas.twin.env import SumoTwinEnv

logger = get_logger(__name__)

ScenarioType = Literal["accident", "closure", "construction", "weather"]


class ScenarioSpec(BaseModel):
    """One declarative disruption: what happens to which edge(s), when, for how long."""

    name: str
    type: ScenarioType
    edge_id: str
    start_s: int = Field(ge=0)
    duration_s: int = Field(gt=0)
    speed_factor: float | None = Field(default=None, gt=0, lt=1)
    """Required for accident/construction/weather (fraction of normal speed during the
    disruption); must be omitted for closure, which always fully blocks the edge."""
    lane_indices: list[int] | None = None
    """construction only: which of edge_id's lanes to fully block (a real lane drop,
    not the whole edge). Required and non-empty for construction; must be omitted for
    every other type."""
    area_edge_ids: list[str] = Field(default_factory=list)
    """weather only: additional edges (beyond edge_id) the same speed_factor derate
    applies to at once -- the "area/global" part of a weather event. Must be empty for
    every other type."""

    @model_validator(mode="after")
    def _validate_fields_match_type(self) -> ScenarioSpec:
        if self.type == "accident":
            if self.speed_factor is None:
                raise ValueError("speed_factor is required when type='accident'")
            if self.lane_indices is not None:
                raise ValueError("lane_indices must not be set when type='accident'")
            if self.area_edge_ids:
                raise ValueError("area_edge_ids must not be set when type='accident'")
        elif self.type == "closure":
            if self.speed_factor is not None:
                raise ValueError("speed_factor must not be set when type='closure'")
            if self.lane_indices is not None:
                raise ValueError("lane_indices must not be set when type='closure'")
            if self.area_edge_ids:
                raise ValueError("area_edge_ids must not be set when type='closure'")
        elif self.type == "construction":
            if self.speed_factor is None:
                raise ValueError("speed_factor is required when type='construction'")
            if not self.lane_indices:
                raise ValueError("lane_indices is required and non-empty when type='construction'")
            if self.area_edge_ids:
                raise ValueError("area_edge_ids must not be set when type='construction'")
        elif self.type == "weather":
            if self.speed_factor is None:
                raise ValueError("speed_factor is required when type='weather'")
            if self.lane_indices is not None:
                raise ValueError("lane_indices must not be set when type='weather'")
        return self


class ScenarioSet(BaseModel):
    network: str
    scenarios: list[ScenarioSpec]


@dataclass
class _AppliedScenario:
    spec: ScenarioSpec
    saved_allowed: dict[str, tuple[str, ...]] = field(default_factory=dict)
    saved_max_speed: dict[str, float] = field(default_factory=dict)


class ScenarioController:
    """Steps a SumoTwinEnv while automatically applying/reverting a set of scenarios at
    their configured simulation times.
    """

    def __init__(self, env: SumoTwinEnv, specs: list[ScenarioSpec]) -> None:
        self._env = env
        self._specs = specs
        self._active: dict[str, _AppliedScenario] = {}

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            self._env.step()
            self._sync()

    def sync(self) -> None:
        """Applies/reverts scenarios for the CURRENT sim time without stepping the
        twin itself -- for callers that already advance a different env type this
        controller doesn't step directly (Week 11's evaluation runner drives a
        `MultiAgentSignalEnv`, whose `step(actions)` needs per-junction actions this
        controller has no business choosing). Call this once per real simulation step
        already taken elsewhere; `step()` above remains the right choice for anything
        that owns a plain `SumoTwinEnv` directly.
        """
        self._sync()

    def _sync(self) -> None:
        t = self._env.sim_time
        for spec in self._specs:
            is_in_window = spec.start_s <= t < spec.start_s + spec.duration_s
            if is_in_window and spec.name not in self._active:
                self._apply(spec)
            elif not is_in_window and spec.name in self._active:
                self._revert(spec)

    def _lanes_of(self, edge_id: str) -> list[str]:
        conn: Any = self._env.connection
        num_lanes: int = conn.edge.getLaneNumber(edge_id)
        return [f"{edge_id}_{i}" for i in range(num_lanes)]

    def _apply(self, spec: ScenarioSpec) -> None:
        conn: Any = self._env.connection
        applied = _AppliedScenario(spec=spec)

        if spec.type == "closure":
            for lane_id in self._lanes_of(spec.edge_id):
                applied.saved_allowed[lane_id] = tuple(conn.lane.getAllowed(lane_id))
                conn.lane.setAllowed(lane_id, [])

        elif spec.type == "accident":
            assert spec.speed_factor is not None  # enforced by ScenarioSpec validation
            for lane_id in self._lanes_of(spec.edge_id):
                original_speed = float(conn.lane.getMaxSpeed(lane_id))
                applied.saved_max_speed[lane_id] = original_speed
                conn.lane.setMaxSpeed(lane_id, original_speed * spec.speed_factor)

        elif spec.type == "construction":
            assert spec.speed_factor is not None
            assert spec.lane_indices is not None
            edge_lanes = self._lanes_of(spec.edge_id)
            dropped = {f"{spec.edge_id}_{i}" for i in spec.lane_indices}
            for lane_id in dropped:
                applied.saved_allowed[lane_id] = tuple(conn.lane.getAllowed(lane_id))
                conn.lane.setAllowed(lane_id, [])
            for lane_id in edge_lanes:
                if lane_id in dropped:
                    continue  # already fully blocked -- a speed limit on top is moot
                original_speed = float(conn.lane.getMaxSpeed(lane_id))
                applied.saved_max_speed[lane_id] = original_speed
                conn.lane.setMaxSpeed(lane_id, original_speed * spec.speed_factor)

        elif spec.type == "weather":
            assert spec.speed_factor is not None
            for edge_id in [spec.edge_id, *spec.area_edge_ids]:
                for lane_id in self._lanes_of(edge_id):
                    original_speed = float(conn.lane.getMaxSpeed(lane_id))
                    applied.saved_max_speed[lane_id] = original_speed
                    conn.lane.setMaxSpeed(lane_id, original_speed * spec.speed_factor)

        self._active[spec.name] = applied
        logger.info(
            "scenario.applied",
            name=spec.name,
            type=spec.type,
            edge_id=spec.edge_id,
            sim_time=self._env.sim_time,
        )

    def _revert(self, spec: ScenarioSpec) -> None:
        conn: Any = self._env.connection
        applied = self._active.pop(spec.name)
        for lane_id, allowed in applied.saved_allowed.items():
            conn.lane.setAllowed(lane_id, list(allowed))
        for lane_id, speed in applied.saved_max_speed.items():
            conn.lane.setMaxSpeed(lane_id, speed)
        logger.info(
            "scenario.reverted",
            name=spec.name,
            type=spec.type,
            edge_id=spec.edge_id,
            sim_time=self._env.sim_time,
        )

    @property
    def active_scenario_names(self) -> frozenset[str]:
        return frozenset(self._active.keys())
