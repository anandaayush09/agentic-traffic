#!/usr/bin/env python
"""Generate the Week 1 placeholder SUMO network + demand used to smoke-test the digital
twin wrapper (`atlas.twin.SumoTwinEnv`) before the real, audited OpenStreetMap district
exists (that is Week 2's job -- see docs/design/08_data_strategy.md section 8.1).

Produces a 3x3 grid network with 4 signalized junctions via SUMO's `netgenerate`, and a
random demand set via `randomTrips.py`, both bundled with the `eclipse-sumo` package
(no manual SUMO install required -- see atlas.common.sumo_locate).

Everything this script produces is regenerable from this file's parameters and a seed
(Part 8.6's synthetic-data policy: every synthetic artifact must be regenerable from
versioned config + seed), so the output directory is gitignored.

Usage:
    python scripts/generate_sandbox_network.py [--force]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atlas.common.config import project_root  # noqa: E402
from atlas.common.logging import configure_logging, get_logger  # noqa: E402
from atlas.common.sumo_locate import get_binary_path, get_tool_path  # noqa: E402

configure_logging()
logger = get_logger(__name__)

OUTPUT_DIR = project_root() / "data" / "networks" / "sandbox"
NET_FILE = OUTPUT_DIR / "net.net.xml"
ROUTE_FILE = OUTPUT_DIR / "routes.rou.xml"

# Grid + traffic-light parameters, matched to configs/network/sandbox.yaml's
# `traffic_light_ids`. Kept as named constants (not CLI flags) because this network is a
# fixed, versioned fixture -- changing its shape means updating the config that describes
# it, not passing a different flag at generation time.
GRID_X, GRID_Y = 3, 3
LANE_NUMBER = 1
TLS_JUNCTIONS = ["A1", "B0", "B1", "B2"]
SEED = 42
NUM_TRIP_END_TIME_S = 200
TRIP_PERIOD_S = 2  # average seconds between successive trip insertions


def generate(force: bool = False) -> None:
    if NET_FILE.is_file() and ROUTE_FILE.is_file() and not force:
        logger.info("sandbox_network.skip_existing", net_file=str(NET_FILE))
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    netgenerate_cmd = [
        str(get_binary_path("netgenerate")),
        "--grid",
        f"--grid.x-number={GRID_X}",
        f"--grid.y-number={GRID_Y}",
        f"--default.lanenumber={LANE_NUMBER}",
        f"--tls.set={','.join(TLS_JUNCTIONS)}",
        f"--output-file={NET_FILE}",
    ]
    logger.info("sandbox_network.generating_net", cmd=" ".join(netgenerate_cmd))
    _run(netgenerate_cmd)

    # -o pins randomTrips.py's intermediate trip file inside OUTPUT_DIR; without it, the
    # tool writes "trips.trips.xml" into the current working directory instead, which
    # pollutes the repo root when this script is invoked from there.
    random_trips_cmd = [
        sys.executable,
        str(get_tool_path("randomTrips.py")),
        "-n",
        str(NET_FILE),
        "-r",
        str(ROUTE_FILE),
        "-o",
        str(OUTPUT_DIR / "trips.trips.xml"),
        "-e",
        str(NUM_TRIP_END_TIME_S),
        "-p",
        str(TRIP_PERIOD_S),
        "--seed",
        str(SEED),
        "--validate",
    ]
    logger.info("sandbox_network.generating_routes", cmd=" ".join(random_trips_cmd))
    _run(random_trips_cmd)

    if not NET_FILE.is_file() or not ROUTE_FILE.is_file():
        raise RuntimeError("Generation reported success but expected output files are missing.")

    logger.info(
        "sandbox_network.done",
        net_file=str(NET_FILE),
        route_file=str(ROUTE_FILE),
    )


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        logger.error("sandbox_network.command_failed", stdout=result.stdout, stderr=result.stderr)
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="Regenerate even if output files already exist."
    )
    args = parser.parse_args()
    generate(force=args.force)


if __name__ == "__main__":
    main()
