"""Validates the real Week 2 district network (Midtown Manhattan). Unlike the sandbox
network's generator (pure local SUMO tools, no external dependency), this district's raw
OSM extract comes from a live API and is committed to git specifically so tests never
need network access -- `build_network()` only ever reads the committed `raw.osm.xml`
unless `--refetch` is passed explicitly at the CLI, which nothing here does.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from xml.etree import ElementTree

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from atlas.common.config import load_yaml_model, project_root  # noqa: E402
from atlas.common.sumo_locate import get_sumo_home  # noqa: E402
from atlas.twin import SumoTwinEnv, TwinEnvConfig  # noqa: E402

MIN_EXPECTED_TRAFFIC_LIGHTS = 15
MAX_EXPECTED_TRAFFIC_LIGHTS = 30


@pytest.fixture(scope="session")
def district_net_file() -> Path:
    import build_district_network

    build_district_network.build_network()
    return build_district_network.NET_FILE


@pytest.fixture
def district_twin_config(district_net_file: Path, tmp_path: Path) -> TwinEnvConfig:
    # Load the real committed config (exercises it directly, catching drift between the
    # YAML and the actual network) and only override the route file: demand generation is
    # a later Week 2 step, so an empty-but-schema-valid route file is enough here to prove
    # the network itself loads and steps correctly.
    config_path = project_root() / "configs" / "network" / "manhattan_midtown.yaml"
    config = load_yaml_model(config_path, TwinEnvConfig)

    empty_routes = tmp_path / "empty.rou.xml"
    empty_routes.write_text('<?xml version="1.0" encoding="UTF-8"?>\n<routes/>\n', encoding="utf-8")
    # Cleared explicitly: the YAML's additional_files points at the fleet vType defs
    # (configs/fleet/...), generated later by scripts/generate_demand.py. This test uses
    # an empty (no-vehicle) route file and must stay independent of demand/fleet
    # generation having run yet -- it is purely a network-structure check.
    return config.model_copy(
        update={"net_file": district_net_file, "route_file": empty_routes, "additional_files": []}
    )


@pytest.fixture
def district_env(district_twin_config: TwinEnvConfig) -> Iterator[SumoTwinEnv]:
    env = SumoTwinEnv(district_twin_config)
    try:
        yield env
    finally:
        env.close()


class TestDistrictNetworkStructure:
    def test_signalized_junction_count_is_near_target(self, district_net_file: Path) -> None:
        tree = ElementTree.parse(district_net_file)
        junctions = tree.getroot().findall("junction")
        signalized = [j for j in junctions if j.get("type") == "traffic_light"]

        assert MIN_EXPECTED_TRAFFIC_LIGHTS <= len(signalized) <= MAX_EXPECTED_TRAFFIC_LIGHTS, (
            f"Expected roughly 15-30 signalized junctions (design target ~25), "
            f"got {len(signalized)}. If the source bbox changed, update AUDIT.md too."
        )

    def test_network_is_a_single_connected_component(self, district_net_file: Path) -> None:
        netcheck = get_sumo_home() / "tools" / "net" / "netcheck.py"
        result = subprocess.run(
            [sys.executable, str(netcheck), str(district_net_file)],
            capture_output=True,
            text=True,
            check=True,
        )

        assert "Coverage: 100.0%" in result.stdout, (
            "Network has disconnected components (islands) -- see netcheck output:\n"
            + result.stdout
        )


class TestDistrictNetworkSimulation:
    def test_loads_and_reports_configured_traffic_lights(
        self, district_env: SumoTwinEnv, district_twin_config: TwinEnvConfig
    ) -> None:
        district_env.reset()

        reported = set(district_env.get_traffic_light_ids())

        assert set(district_twin_config.traffic_light_ids) == reported

    def test_steps_cleanly_and_reads_junction_state(self, district_env: SumoTwinEnv) -> None:
        district_env.reset()

        district_env.step(n=50)
        states = district_env.get_all_junction_states()

        assert district_env.sim_time == pytest.approx(50.0)
        assert len(states) == len(district_env.config.traffic_light_ids)
        for state in states.values():
            assert len(state.incoming_lanes) > 0
