#!/usr/bin/env python
"""Generates denser demand for the sandbox network's RL ramp-up (Week 3), separate from
the sparse demand scripts/generate_sandbox_network.py produces for the Week 1 twin smoke
test. Two independent realizations (different seeds) -- training and evaluation must not
share the exact same vehicle arrivals, or a "beats fixed-time" result would partly just be
memorization of one specific demand instance rather than a real controller improvement.

Usage:
    python scripts/generate_rl_demand.py [--force]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atlas.common.config import project_root  # noqa: E402
from atlas.common.logging import configure_logging, get_logger  # noqa: E402
from atlas.common.sumo_locate import get_tool_path  # noqa: E402

configure_logging()
logger = get_logger(__name__)

NETWORK_DIR = project_root() / "data" / "networks" / "sandbox"
NET_FILE = NETWORK_DIR / "net.net.xml"

DURATION_S = 1200  # 20 simulated minutes per episode
PERIOD_S = 1.5  # denser than the Week 1 smoke-test demand (period_s=2.0, 200s only)
FRINGE_FACTOR = 5.0

TRAIN_ROUTE_FILE = NETWORK_DIR / "rl_train.rou.xml"
EVAL_ROUTE_FILE = NETWORK_DIR / "rl_eval.rou.xml"


def _generate(route_file: Path, seed: int, force: bool) -> Path:
    if route_file.is_file() and not force:
        logger.info("rl_demand.skip_existing", route_file=str(route_file))
        return route_file
    if not NET_FILE.is_file():
        raise RuntimeError(f"{NET_FILE} not found. Run scripts/generate_sandbox_network.py first.")

    trip_file = route_file.with_suffix(".trips.xml")
    cmd = [
        sys.executable,
        str(get_tool_path("randomTrips.py")),
        "-n",
        str(NET_FILE),
        "-r",
        str(route_file),
        "-o",
        str(trip_file),
        "-e",
        str(DURATION_S),
        "-p",
        str(PERIOD_S),
        "--fringe-factor",
        str(FRINGE_FACTOR),
        "--seed",
        str(seed),
        "--validate",
    ]
    logger.info("rl_demand.generating", route_file=str(route_file), seed=seed)
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        logger.error("rl_demand.generation_failed", stderr=result.stderr)
        raise RuntimeError(f"randomTrips.py failed for {route_file}")
    if not route_file.is_file():
        raise RuntimeError(f"Generation reported success but {route_file} is missing.")
    return route_file


def generate_all(force: bool = False) -> tuple[Path, Path]:
    train = _generate(TRAIN_ROUTE_FILE, seed=201, force=force)
    eval_ = _generate(EVAL_ROUTE_FILE, seed=202, force=force)
    return train, eval_


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    generate_all(force=args.force)


if __name__ == "__main__":
    main()
