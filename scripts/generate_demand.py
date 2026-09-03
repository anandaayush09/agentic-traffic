#!/usr/bin/env python
"""Generate time-of-day demand periods for a district network from a
configs/demand/*.yaml recipe (Week 2 part 2 -- see docs/design/08_data_strategy.md
section 8.2).

For each period, runs `randomTrips.py --validate` (which internally calls `duarouter`,
producing fully-resolved <route edges="..."/> vehicles rather than bare trips -- required
for the turn-direction audit in atlas.twin.demand to have edge sequences to read).

Usage:
    python scripts/generate_demand.py [--config configs/demand/manhattan_midtown.yaml] [--force]
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

configure_logging()
logger = get_logger(__name__)

DEFAULT_CONFIG_PATH = project_root() / "configs" / "demand" / "manhattan_midtown.yaml"


def generate_period(net_file: Path, output_dir: Path, period: DemandPeriod, force: bool) -> Path:
    route_file = output_dir / f"{period.name}.rou.xml"
    trip_file = output_dir / f"{period.name}.trips.xml"
    if route_file.is_file() and not force:
        logger.info("demand.skip_existing", period=period.name, route_file=str(route_file))
        return route_file

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(get_tool_path("randomTrips.py")),
        "-n",
        str(net_file),
        "-r",
        str(route_file),
        "-o",
        str(trip_file),
        "-e",
        str(period.duration_s),
        "-p",
        str(period.period_s),
        "--fringe-factor",
        str(period.fringe_factor),
        "--seed",
        str(period.seed),
        "--validate",
    ]
    logger.info("demand.generating", period=period.name, cmd=" ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        logger.error("demand.generation_failed", period=period.name, stderr=result.stderr)
        raise RuntimeError(f"randomTrips.py failed for period '{period.name}'")

    if not route_file.is_file():
        raise RuntimeError(f"Generation reported success but {route_file} is missing.")
    return route_file


def generate_all(config_path: Path = DEFAULT_CONFIG_PATH, force: bool = False) -> dict[str, Path]:
    demand_config = load_yaml_model(config_path, DemandConfig)
    net_file = project_root() / "data" / "networks" / demand_config.network / "net.net.xml"
    if not net_file.is_file():
        raise RuntimeError(
            f"{net_file} not found. Build the network first, e.g. "
            "`python scripts/build_district_network.py`."
        )
    output_dir = project_root() / "data" / "networks" / demand_config.network / "demand"

    generated = {}
    for period in demand_config.periods:
        generated[period.name] = generate_period(net_file, output_dir, period, force)
    logger.info("demand.done", periods=list(generated.keys()))
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    generate_all(config_path=args.config, force=args.force)


if __name__ == "__main__":
    main()
