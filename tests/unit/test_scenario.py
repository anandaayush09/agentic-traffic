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

    def test_construction_requires_speed_factor(self) -> None:
        with pytest.raises(ValidationError, match="speed_factor is required"):
            ScenarioSpec(
                name="x",
                type="construction",
                edge_id="e1",
                start_s=0,
                duration_s=10,
                lane_indices=[0],
            )

    def test_construction_requires_nonempty_lane_indices(self) -> None:
        with pytest.raises(ValidationError, match="lane_indices is required"):
            ScenarioSpec(
                name="x",
                type="construction",
                edge_id="e1",
                start_s=0,
                duration_s=10,
                speed_factor=0.5,
            )

    def test_valid_construction_spec(self) -> None:
        spec = ScenarioSpec(
            name="x",
            type="construction",
            edge_id="e1",
            start_s=0,
            duration_s=10,
            speed_factor=0.7,
            lane_indices=[0, 1],
        )

        assert spec.lane_indices == [0, 1]

    def test_weather_requires_speed_factor(self) -> None:
        with pytest.raises(ValidationError, match="speed_factor is required"):
            ScenarioSpec(name="x", type="weather", edge_id="e1", start_s=0, duration_s=10)

    def test_weather_forbids_lane_indices(self) -> None:
        with pytest.raises(ValidationError, match="lane_indices must not be set"):
            ScenarioSpec(
                name="x",
                type="weather",
                edge_id="e1",
                start_s=0,
                duration_s=10,
                speed_factor=0.6,
                lane_indices=[0],
            )

    def test_valid_weather_spec_with_area_edges(self) -> None:
        spec = ScenarioSpec(
            name="x",
            type="weather",
            edge_id="e1",
            area_edge_ids=["e2", "e3"],
            start_s=0,
            duration_s=10,
            speed_factor=0.6,
        )

        assert spec.area_edge_ids == ["e2", "e3"]


class TestScenarioConfigLoading:
    def test_loads_all_eight_scenarios(self) -> None:
        config = load_yaml_model(SCENARIO_CONFIG_PATH, ScenarioSet)

        names = {s.name for s in config.scenarios}
        assert names == {
            "midblock_accident",
            "avenue_closure",
            "lane_drop_construction",
            "regional_weather",
            "severe_accident",
            "crossstreet_closure",
            "heavy_construction",
            "severe_weather",
        }


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
    # additional_files: the demand is fleet-mixed, so vehicles reference vType ids that
    # must be resolvable via the generated vtypes file.
    vtypes_file = NET_FILE.parent / "vtypes.add.xml"
    config = TwinEnvConfig(
        name="scenario_test",
        net_file=NET_FILE,
        route_file=night_demand,
        enable_rerouting=True,
        additional_files=[vtypes_file],
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

    def test_construction_drops_only_its_own_lanes_and_derates_the_rest(
        self, scenario_env: SumoTwinEnv
    ) -> None:
        config = load_yaml_model(SCENARIO_CONFIG_PATH, ScenarioSet)
        construction = next(s for s in config.scenarios if s.name == "lane_drop_construction")
        assert construction.lane_indices is not None
        controller = ScenarioController(scenario_env, [construction])
        lanes = _lane_ids(scenario_env, construction.edge_id)
        dropped_lanes = {f"{construction.edge_id}_{i}" for i in construction.lane_indices}
        remaining_lanes = [lid for lid in lanes if lid not in dropped_lanes]
        assert remaining_lanes, "test needs at least one non-dropped lane to be meaningful"
        original_allowed = {
            lid: tuple(scenario_env.connection.lane.getAllowed(lid)) for lid in dropped_lanes
        }
        original_speeds = {
            lid: scenario_env.connection.lane.getMaxSpeed(lid) for lid in remaining_lanes
        }

        controller.step(n=construction.start_s - 1)
        assert "lane_drop_construction" not in controller.active_scenario_names

        controller.step(n=2)
        assert "lane_drop_construction" in controller.active_scenario_names
        for lid in dropped_lanes:
            assert scenario_env.connection.lane.getAllowed(lid) == ()  # fully blocked
        for lid in remaining_lanes:
            assert scenario_env.connection.lane.getMaxSpeed(lid) == pytest.approx(
                original_speeds[lid] * construction.speed_factor
            )

        controller.step(n=construction.duration_s)
        assert "lane_drop_construction" not in controller.active_scenario_names
        for lid in dropped_lanes:
            assert scenario_env.connection.lane.getAllowed(lid) == original_allowed[lid]
        for lid in remaining_lanes:
            assert scenario_env.connection.lane.getMaxSpeed(lid) == pytest.approx(
                original_speeds[lid]
            )

    def test_second_variant_per_class_applies_and_reverts_correctly(
        self, scenario_env: SumoTwinEnv
    ) -> None:
        """The Week 11 close-out second variant per class (severe_accident,
        crossstreet_closure, heavy_construction, severe_weather) -- a different
        corridor segment AND a different severity than the original four, run
        together through one real episode so each one's own apply/revert is checked
        independently at its own [start_s, start_s+duration_s) window, the same way
        the design's real multi-scenario suite would exercise them.
        """
        config = load_yaml_model(SCENARIO_CONFIG_PATH, ScenarioSet)
        second_variants = {
            "severe_accident": next(s for s in config.scenarios if s.name == "severe_accident"),
            "crossstreet_closure": next(
                s for s in config.scenarios if s.name == "crossstreet_closure"
            ),
            "heavy_construction": next(
                s for s in config.scenarios if s.name == "heavy_construction"
            ),
            "severe_weather": next(s for s in config.scenarios if s.name == "severe_weather"),
        }
        controller = ScenarioController(scenario_env, list(second_variants.values()))

        accident = second_variants["severe_accident"]
        accident_lanes = _lane_ids(scenario_env, accident.edge_id)
        accident_original_speeds = {
            lid: scenario_env.connection.lane.getMaxSpeed(lid) for lid in accident_lanes
        }

        closure = second_variants["crossstreet_closure"]
        closure_lanes = _lane_ids(scenario_env, closure.edge_id)
        closure_original_allowed = {
            lid: tuple(scenario_env.connection.lane.getAllowed(lid)) for lid in closure_lanes
        }

        construction = second_variants["heavy_construction"]
        assert construction.lane_indices is not None
        construction_dropped = {f"{construction.edge_id}_{i}" for i in construction.lane_indices}
        construction_remaining = [
            lid
            for lid in _lane_ids(scenario_env, construction.edge_id)
            if lid not in construction_dropped
        ]
        assert construction_remaining, "heavy_construction must leave at least one open lane"
        construction_original_speeds = {
            lid: scenario_env.connection.lane.getMaxSpeed(lid) for lid in construction_remaining
        }

        weather = second_variants["severe_weather"]
        weather_edges = [weather.edge_id, *weather.area_edge_ids]
        weather_lanes_by_edge = {e: _lane_ids(scenario_env, e) for e in weather_edges}
        weather_original_speeds = {
            lid: scenario_env.connection.lane.getMaxSpeed(lid)
            for lanes in weather_lanes_by_edge.values()
            for lid in lanes
        }

        # Advance through every window in one pass, checking each scenario is active
        # only during its own window and fully reverted immediately after.
        controller.step(n=accident.start_s + 2)
        assert "severe_accident" in controller.active_scenario_names
        for lid in accident_lanes:
            assert scenario_env.connection.lane.getMaxSpeed(lid) == pytest.approx(
                accident_original_speeds[lid] * accident.speed_factor
            )

        controller.step(n=accident.duration_s)
        assert "severe_accident" not in controller.active_scenario_names
        for lid in accident_lanes:
            assert scenario_env.connection.lane.getMaxSpeed(lid) == pytest.approx(
                accident_original_speeds[lid]
            )

        controller.step(n=int(closure.start_s - scenario_env.sim_time) + 2)
        assert "crossstreet_closure" in controller.active_scenario_names
        for lid in closure_lanes:
            assert scenario_env.connection.lane.getAllowed(lid) == ()

        controller.step(n=closure.duration_s)
        assert "crossstreet_closure" not in controller.active_scenario_names
        for lid in closure_lanes:
            assert scenario_env.connection.lane.getAllowed(lid) == closure_original_allowed[lid]

        controller.step(n=int(construction.start_s - scenario_env.sim_time) + 2)
        assert "heavy_construction" in controller.active_scenario_names
        for lid in construction_dropped:
            assert scenario_env.connection.lane.getAllowed(lid) == ()
        for lid in construction_remaining:
            assert scenario_env.connection.lane.getMaxSpeed(lid) == pytest.approx(
                construction_original_speeds[lid] * construction.speed_factor
            )

        controller.step(n=construction.duration_s)
        assert "heavy_construction" not in controller.active_scenario_names
        for lid in construction_remaining:
            assert scenario_env.connection.lane.getMaxSpeed(lid) == pytest.approx(
                construction_original_speeds[lid]
            )

        controller.step(n=int(weather.start_s - scenario_env.sim_time) + 2)
        assert "severe_weather" in controller.active_scenario_names
        for lanes in weather_lanes_by_edge.values():
            for lid in lanes:
                assert scenario_env.connection.lane.getMaxSpeed(lid) == pytest.approx(
                    weather_original_speeds[lid] * weather.speed_factor
                )

        controller.step(n=weather.duration_s)
        assert "severe_weather" not in controller.active_scenario_names
        for lanes in weather_lanes_by_edge.values():
            for lid in lanes:
                assert scenario_env.connection.lane.getMaxSpeed(lid) == pytest.approx(
                    weather_original_speeds[lid]
                )

    def test_weather_derates_every_area_edge_at_once(self, scenario_env: SumoTwinEnv) -> None:
        config = load_yaml_model(SCENARIO_CONFIG_PATH, ScenarioSet)
        weather = next(s for s in config.scenarios if s.name == "regional_weather")
        controller = ScenarioController(scenario_env, [weather])
        all_edges = [weather.edge_id, *weather.area_edge_ids]
        assert len(all_edges) > 1, "test needs a real multi-edge area to be meaningful"
        lanes_by_edge = {edge_id: _lane_ids(scenario_env, edge_id) for edge_id in all_edges}
        original_speeds = {
            lid: scenario_env.connection.lane.getMaxSpeed(lid)
            for lanes in lanes_by_edge.values()
            for lid in lanes
        }

        controller.step(n=weather.start_s - 1)
        assert "regional_weather" not in controller.active_scenario_names

        controller.step(n=2)
        assert "regional_weather" in controller.active_scenario_names
        for lanes in lanes_by_edge.values():
            for lid in lanes:
                assert scenario_env.connection.lane.getMaxSpeed(lid) == pytest.approx(
                    original_speeds[lid] * weather.speed_factor
                )

        controller.step(n=weather.duration_s)
        assert "regional_weather" not in controller.active_scenario_names
        for lanes in lanes_by_edge.values():
            for lid in lanes:
                assert scenario_env.connection.lane.getMaxSpeed(lid) == pytest.approx(
                    original_speeds[lid]
                )
