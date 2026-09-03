"""Validates the scenario engine: spec validation, config loading, and (against the real
Manhattan network + night demand) that accident/closure scenarios apply the correct lane
mutation exactly during their time window and revert exactly afterward.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError

from atlas.common.config import load_yaml_model, project_root
from atlas.twin import ScenarioController, ScenarioSet, ScenarioSpec, SumoTwinEnv, TwinEnvConfig

NET_FILE = project_root() / "data" / "networks" / "manhattan_midtown" / "net.net.xml"
NIGHT_ROUTE_FILE = (
    project_root() / "data" / "networks" / "manhattan_midtown" / "demand" / "night.rou.xml"
)
SCENARIO_CONFIG_PATH = project_root() / "configs" / "scenarios" / "manhattan_midtown.yaml"


class TestScenarioSpecValidation:
    def test_accident_requires_speed_factor(self) -> None:
        with pytest.raises(ValidationError, match="speed_factor is required"):
            ScenarioSpec(name="x", type="accident", edge_id="e1", start_s=0, duration_s=10)

    def test_closure_forbids_speed_factor(self) -> None:
        with pytest.raises(ValidationError, match="must not be set"):
            ScenarioSpec(
                name="x", type="closure", edge_id="e1", start_s=0, duration_s=10, speed_factor=0.5
            )

    def test_speed_factor_must_be_fractional(self) -> None:
        with pytest.raises(ValidationError):
            ScenarioSpec(
                name="x", type="accident", edge_id="e1", start_s=0, duration_s=10, speed_factor=1.5
            )

    def test_valid_accident_spec(self) -> None:
        spec = ScenarioSpec(
            name="x", type="accident", edge_id="e1", start_s=0, duration_s=10, speed_factor=0.3
        )

        assert spec.speed_factor == 0.3


class TestScenarioConfigLoading:
    def test_loads_two_scenarios(self) -> None:
        config = load_yaml_model(SCENARIO_CONFIG_PATH, ScenarioSet)

        names = {s.name for s in config.scenarios}
        assert names == {"midblock_accident", "avenue_closure"}


@pytest.fixture(scope="session")
def night_demand() -> Path:
    import sys

    sys.path.insert(0, str(project_root() / "scripts"))
    import build_district_network
    import generate_demand

    build_district_network.build_network()
    generate_demand.generate_all()
    return NIGHT_ROUTE_FILE


@pytest.fixture
def scenario_env(night_demand: Path) -> Iterator[SumoTwinEnv]:
    # enable_rerouting: a closure can invalidate a vehicle's pre-computed route; without
    # this, SUMO's --ignore-route-errors baseline just drops every such vehicle instead of
    # routing it around the closure (see the Week 2 module writeup for the full story).
    config = TwinEnvConfig(
        name="scenario_test", net_file=NET_FILE, route_file=night_demand, enable_rerouting=True
    )
    env = SumoTwinEnv(config)
    env.reset()
    try:
        yield env
    finally:
        env.close()


def _lane_ids(env: SumoTwinEnv, edge_id: str) -> list[str]:
    num_lanes = env.connection.edge.getLaneNumber(edge_id)
    return [f"{edge_id}_{i}" for i in range(num_lanes)]


class TestScenarioControllerLifecycle:
    def test_accident_reduces_and_restores_speed(self, scenario_env: SumoTwinEnv) -> None:
        config = load_yaml_model(SCENARIO_CONFIG_PATH, ScenarioSet)
        accident = next(s for s in config.scenarios if s.name == "midblock_accident")
        controller = ScenarioController(scenario_env, [accident])
        lanes = _lane_ids(scenario_env, accident.edge_id)
        original_speeds = {lid: scenario_env.connection.lane.getMaxSpeed(lid) for lid in lanes}

        controller.step(n=accident.start_s - 1)
        assert "midblock_accident" not in controller.active_scenario_names
        for lid in lanes:
            assert scenario_env.connection.lane.getMaxSpeed(lid) == pytest.approx(
                original_speeds[lid]
            )

        controller.step(n=2)
        assert "midblock_accident" in controller.active_scenario_names
        for lid in lanes:
            assert scenario_env.connection.lane.getMaxSpeed(lid) == pytest.approx(
                original_speeds[lid] * accident.speed_factor
            )

        controller.step(n=accident.duration_s)
        assert "midblock_accident" not in controller.active_scenario_names
        for lid in lanes:
            assert scenario_env.connection.lane.getMaxSpeed(lid) == pytest.approx(
                original_speeds[lid]
            )

    def test_closure_blocks_and_restores_lane(self, scenario_env: SumoTwinEnv) -> None:
        config = load_yaml_model(SCENARIO_CONFIG_PATH, ScenarioSet)
        closure = next(s for s in config.scenarios if s.name == "avenue_closure")
        controller = ScenarioController(scenario_env, [closure])
        lanes = _lane_ids(scenario_env, closure.edge_id)
        original_allowed = {
            lid: tuple(scenario_env.connection.lane.getAllowed(lid)) for lid in lanes
        }

        controller.step(n=closure.start_s - 1)
        for lid in lanes:
            assert scenario_env.connection.lane.getAllowed(lid) == original_allowed[lid]

        controller.step(n=2)
        assert "avenue_closure" in controller.active_scenario_names
        for lid in lanes:
            assert scenario_env.connection.lane.getAllowed(lid) == ()

        controller.step(n=closure.duration_s)
        assert "avenue_closure" not in controller.active_scenario_names
        for lid in lanes:
            assert scenario_env.connection.lane.getAllowed(lid) == original_allowed[lid]
