"""Validates generated demand against plausibility invariants (docs/design/
08_data_strategy.md section 8.2) rather than real-world ground truth, which doesn't exist
for a synthetic district: turn-direction sanity (not degenerate) and a congestion proxy
that must order peak > off-peak periods and stay within plausible bounds.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from atlas.common.config import load_yaml_model, project_root
from atlas.twin import SumoTwinEnv, TwinEnvConfig
from atlas.twin.demand import DemandConfig, parse_connection_directions, turn_direction_counts

DEMAND_CONFIG_PATH = project_root() / "configs" / "demand" / "manhattan_midtown.yaml"
NET_FILE = project_root() / "data" / "networks" / "manhattan_midtown" / "net.net.xml"
DEMAND_DIR = project_root() / "data" / "networks" / "manhattan_midtown" / "demand"

SAMPLE_INTERVAL_STEPS = 60


@pytest.fixture(scope="session")
def demand_config() -> DemandConfig:
    return load_yaml_model(DEMAND_CONFIG_PATH, DemandConfig)


@pytest.fixture(scope="session")
def generated_demand(demand_config: DemandConfig) -> dict[str, Path]:
    # scripts/ is already on sys.path via tests/conftest.py.
    import build_district_network
    import generate_demand

    build_district_network.build_network()
    return generate_demand.generate_all(demand_config_path=DEMAND_CONFIG_PATH)


def _mean_speed_ratio(
    route_file: Path, duration_s: int, sample_interval: int = SAMPLE_INTERVAL_STEPS
) -> float:
    """Runs the given demand through a fresh SumoTwinEnv-managed simulation and returns
    the mean (observed speed / allowed speed) across all present vehicles, sampled every
    `sample_interval` steps. 1.0 = free flow; lower = more congested. Reaches into
    `env.connection` (rather than extending SumoTwinEnv's public API) because network-wide
    vehicle speed sampling is a one-off validation concern, not a twin-wrapper
    responsibility -- see the Week 2 module writeup for the rationale.
    """
    # additional_files: the demand is fleet-mixed (configs/fleet/manhattan_midtown.yaml),
    # so vehicles reference vType ids that must be resolvable via the generated vtypes file.
    vtypes_file = NET_FILE.parent / "vtypes.add.xml"
    config = TwinEnvConfig(
        name="congestion_probe",
        net_file=NET_FILE,
        route_file=route_file,
        additional_files=[vtypes_file],
    )
    env = SumoTwinEnv(config)
    ratios: list[float] = []
    try:
        env.reset()
        conn = env.connection

        duration_steps = int(duration_s / env.config.step_length_s)
        for step in range(duration_steps):
            env.step()
            if step % sample_interval == 0:
                for vehicle_id in conn.vehicle.getIDList():
                    speed = conn.vehicle.getSpeed(vehicle_id)
                    allowed = conn.vehicle.getAllowedSpeed(vehicle_id)
                    if allowed > 0:
                        ratios.append(speed / allowed)
    finally:
        env.close()

    if not ratios:
        raise RuntimeError(f"No vehicles observed while sampling {route_file}")
    return sum(ratios) / len(ratios)


class TestDemandConfig:
    def test_loads_four_periods(self, demand_config: DemandConfig) -> None:
        names = {p.name for p in demand_config.periods}

        assert names == {"night", "midday", "am_peak", "pm_peak"}

    def test_rejects_nonpositive_duration(self) -> None:
        from atlas.twin.demand import DemandPeriod

        with pytest.raises(ValidationError):
            DemandPeriod(name="bad", duration_s=0, period_s=1.0, fringe_factor=1.0, seed=1)


class TestDemandVolumeOrdering:
    def test_peak_periods_have_more_vehicles_than_night(
        self, generated_demand: dict[str, Path]
    ) -> None:
        def vehicle_count(route_file: Path) -> int:
            return route_file.read_text(encoding="utf-8").count("<vehicle ")

        night = vehicle_count(generated_demand["night"])
        midday = vehicle_count(generated_demand["midday"])
        am_peak = vehicle_count(generated_demand["am_peak"])
        pm_peak = vehicle_count(generated_demand["pm_peak"])

        assert night < midday < am_peak
        assert night < midday < pm_peak


class TestTurnDirectionAudit:
    def test_connection_directions_cover_all_codes(self, generated_demand: dict[str, Path]) -> None:
        directions = parse_connection_directions(NET_FILE)

        assert len(directions) > 0
        assert set(directions.values()) <= {"s", "l", "r", "t"}

    def test_pm_peak_turn_mix_is_not_degenerate(self, generated_demand: dict[str, Path]) -> None:
        directions = parse_connection_directions(NET_FILE)
        counts = turn_direction_counts(generated_demand["pm_peak"], directions)
        total = sum(counts.values())

        assert total > 0
        straight_fraction = counts.get("s", 0) / total
        turn_fraction = counts.get("t", 0) / total
        # Through-traffic (fringe_factor=5) should dominate but not be exclusive, and
        # U-turns should be rare -- a real district looks like this; a broken network or a
        # demand-generation bug tends to produce something degenerate (all-straight,
        # all-turn, or nothing at all).
        assert 0.3 < straight_fraction < 0.9
        assert turn_fraction < 0.15
        assert counts.get("l", 0) > 0
        assert counts.get("r", 0) > 0


def _duration_for(demand_config: DemandConfig, period_name: str) -> int:
    return next(p.duration_s for p in demand_config.periods if p.name == period_name)


class TestCongestionProxy:
    """Note on the absolute scale: even the `night` period's ratio comes out to ~0.38, not
    close to 1.0 (free flow). That is a real property of this district, not a bug -- with
    a signalized junction roughly every block, most of a vehicle's average delay is
    waiting at red lights, not queueing behind other traffic, so the speed/allowed-speed
    ratio stays well below 1.0 even at very light demand. The invariant this project can
    actually assert (no real ground-truth counts exist to calibrate an absolute target
    against -- see docs/design/08_data_strategy.md section 8.2) is *relative*: the ratio
    must fall monotonically as demand rises, and never collapse to near-total gridlock.
    """

    def test_congestion_ratio_falls_monotonically_with_demand(
        self, generated_demand: dict[str, Path], demand_config: DemandConfig
    ) -> None:
        ratios = {
            name: _mean_speed_ratio(generated_demand[name], _duration_for(demand_config, name))
            for name in ("night", "midday", "am_peak", "pm_peak")
        }

        assert ratios["night"] > ratios["midday"] > ratios["am_peak"] > ratios["pm_peak"]
        for ratio in ratios.values():
            assert 0.1 < ratio <= 1.0, f"ratio {ratio} outside plausible [0.1, 1.0] band"
