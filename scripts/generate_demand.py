#!/usr/bin/env python
"""Generate time-of-day, fleet-mixed demand for a district network from a
configs/demand/*.yaml period recipe + a configs/fleet/*.yaml vehicle-type mix (Week 2
parts 2-3 -- see docs/design/08_data_strategy.md sections 8.2).

For each period, runs one `randomTrips.py --validate` per fleet component (each
referencing a shared vType via `--trip-attributes 'type="..."'`, scaled by
`period_s / fraction` -- see atlas.twin.fleet for why `--vehicle-class` can't be combined
with a type= attribute), then merges the per-component route files into one depart-time-
sorted route file for the period. `--validate` internally calls `duarouter`, producing
fully-resolved <route edges="..."/> vehicles rather than bare trips -- required for the
turn-direction audit in atlas.twin.demand to have edge sequences to read.

Usage:
    python scripts/generate_demand.py [--force]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atlas.common.config import load_yaml_model, project_root  # noqa: E402
from atlas.common.logging import configure_logging, get_logger  # noqa: E402
from atlas.common.sumo_locate import get_tool_path  # noqa: E402
from atlas.twin.demand import DemandConfig, DemandPeriod  # noqa: E402
from atlas.twin.fleet import (  # noqa: E402
    FleetConfig,
    merge_route_files,
    write_vtypes_additional_file,
)

configure_logging()
logger = get_logger(__name__)

DEFAULT_DEMAND_CONFIG_PATH = project_root() / "configs" / "demand" / "manhattan_midtown.yaml"
DEFAULT_FLEET_CONFIG_PATH = project_root() / "configs" / "fleet" / "manhattan_midtown.yaml"


def _generate_component_trips(
    net_file: Path,
    vtypes_file: Path,
    work_dir: Path,
    period: DemandPeriod,
    vtype_id: str,
    fraction: float,
    force: bool,
) -> Path:
    route_file = work_dir / f"{period.name}_{vtype_id}.rou.xml"
    trip_file = work_dir / f"{period.name}_{vtype_id}.trips.xml"
    if route_file.is_file() and not force:
        return route_file

    work_dir.mkdir(parents=True, exist_ok=True)
    component_period_s = period.period_s / fraction
    cmd = [
        sys.executable,
        str(get_tool_path("randomTrips.py")),
        "-n",
        str(net_file),
        "-a",
        str(vtypes_file),
        "-r",
        str(route_file),
        "-o",
        str(trip_file),
        "-e",
        str(period.duration_s),
        "-p",
        str(component_period_s),
        "--fringe-factor",
        str(period.fringe_factor),
        "--seed",
        str(period.seed),
        "--prefix",
        f"{period.name}_{vtype_id}_",
        "--trip-attributes",
        f'type="{vtype_id}"',
        "--validate",
    ]
    logger.info(
        "demand.generating_component",
        period=period.name,
        vtype_id=vtype_id,
        component_period_s=component_period_s,
    )
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        logger.error(
            "demand.component_generation_failed",
            period=period.name,
            vtype_id=vtype_id,
            stderr=result.stderr,
        )
        raise RuntimeError(f"randomTrips.py failed for period '{period.name}', type '{vtype_id}'")
    if not route_file.is_file():
        raise RuntimeError(f"Generation reported success but {route_file} is missing.")
    return route_file


def generate_period(
    net_file: Path,
    vtypes_file: Path,
    output_dir: Path,
    work_dir: Path,
    period: DemandPeriod,
    fleet_config: FleetConfig,
    force: bool,
) -> Path:
    merged_route_file = output_dir / f"{period.name}.rou.xml"
    if merged_route_file.is_file() and not force:
        logger.info("demand.skip_existing", period=period.name, route_file=str(merged_route_file))
        return merged_route_file

    component_files = [
        _generate_component_trips(
            net_file, vtypes_file, work_dir, period, component.vtype_id, component.fraction, force
        )
        for component in fleet_config.components
    ]
    merge_route_files(component_files, merged_route_file)
    logger.info(
        "demand.period_merged",
        period=period.name,
        components=[c.vtype_id for c in fleet_config.components],
        route_file=str(merged_route_file),
    )
    return merged_route_file


def generate_all(
    demand_config_path: Path = DEFAULT_DEMAND_CONFIG_PATH,
    fleet_config_path: Path = DEFAULT_FLEET_CONFIG_PATH,
    force: bool = False,
) -> dict[str, Path]:
    demand_config = load_yaml_model(demand_config_path, DemandConfig)
    fleet_config = load_yaml_model(fleet_config_path, FleetConfig)
    if demand_config.network != fleet_config.network:
        raise RuntimeError(
            f"demand config network '{demand_config.network}' != "
            f"fleet config network '{fleet_config.network}'"
        )

    network_dir = project_root() / "data" / "networks" / demand_config.network
    net_file = network_dir / "net.net.xml"
    if not net_file.is_file():
        raise RuntimeError(
            f"{net_file} not found. Build the network first, e.g. "
            "`python scripts/build_district_network.py`."
        )

    vtypes_file = network_dir / "vtypes.add.xml"
    write_vtypes_additional_file(fleet_config, vtypes_file)

    output_dir = network_dir / "demand"
    work_dir = network_dir / "demand" / "_components"

    generated = {
        period.name: generate_period(
            net_file, vtypes_file, output_dir, work_dir, period, fleet_config, force
        )
        for period in demand_config.periods
    }
    logger.info("demand.done", periods=list(generated.keys()))
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demand-config", type=Path, default=DEFAULT_DEMAND_CONFIG_PATH)
    parser.add_argument("--fleet-config", type=Path, default=DEFAULT_FLEET_CONFIG_PATH)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    generate_all(
        demand_config_path=args.demand_config, fleet_config_path=args.fleet_config, force=args.force
    )


if __name__ == "__main__":
    main()
