"""Scenario engine v1: declarative accident/closure disruptions applied to a running
digital twin (docs/design/08_data_strategy.md section 8.3). Rush-hour is already covered
by the am_peak/pm_peak demand periods (atlas.twin.demand), so it is not a scenario type
here.

Mechanism (verified against libsumo directly before writing this):
- **closure**: `lane.setAllowed(lane_id, [])` on every lane of the target edge -- sets
  the allowed-vehicle-class list to empty, blocking all classes. NOT the same as clearing
  a restriction; reverting requires restoring the lane's original `getAllowed()` tuple.
- **accident**: `lane.setMaxSpeed(lane_id, original_speed * speed_factor)` on every lane
  of the target edge -- a slowdown rather than a full block, distinguishing it from a
  closure. Reverting restores the original max speed.

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

ScenarioType = Literal["accident", "closure"]


class ScenarioSpec(BaseModel):
    """One declarative disruption: what happens to which edge, when, for how long."""

    name: str
    type: ScenarioType
    edge_id: str
    start_s: int = Field(ge=0)
    duration_s: int = Field(gt=0)
    speed_factor: float | None = Field(default=None, gt=0, lt=1)
    """Required for type="accident" (fraction of normal speed during the disruption);
    must be omitted for type="closure", which always fully blocks the edge."""

    @model_validator(mode="after")
    def _validate_speed_factor(self) -> ScenarioSpec:
        if self.type == "accident" and self.speed_factor is None:
            raise ValueError("speed_factor is required when type='accident'")
        if self.type == "closure" and self.speed_factor is not None:
            raise ValueError("speed_factor must not be set when type='closure'")
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
        for lane_id in self._lanes_of(spec.edge_id):
            if spec.type == "closure":
                applied.saved_allowed[lane_id] = tuple(conn.lane.getAllowed(lane_id))
                conn.lane.setAllowed(lane_id, [])
            else:  # accident
                original_speed = float(conn.lane.getMaxSpeed(lane_id))
                applied.saved_max_speed[lane_id] = original_speed
                assert spec.speed_factor is not None  # enforced by ScenarioSpec validation
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
