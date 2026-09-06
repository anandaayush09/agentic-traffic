#!/usr/bin/env python
"""Trains the from-scratch MAPPO trainer on the FULL manhattan_midtown district (all 22
signalized junctions) -- curriculum stage 3 of docs/design/06_ai_pipeline.md section 6.1
("(1) single junction... (2) 1 arterial corridor (5 junctions); (3) full district,
calibrated demand"), the Week 6 roadmap deliverable ("curriculum stage 3 (full district)
training launched... checkpoint-resumable").

Same env/network code as the Week 5 corridor script (scripts/train_mappo_corridor.py),
just a longer `tls_ids` list -- this IS the curriculum recipe: MultiAgentSignalEnv and
the training loop (atlas.marl.training.run_training) don't change, only which junctions
are under agent control. tls_ids/max_lanes below are the same set already measured and
used by scripts/benchmark_ray_rollout.py's Week 4 throughput benchmark.

**Honest scope note**: docs/design/06_ai_pipeline.md's own throughput math says MAPPO
convergence at this scale needs ~5-15M env-steps, "an overnight-to-weekend session" on
Kaggle -- not something this local Windows dev machine is meant to run to convergence.
This script is verified end-to-end here with a short run (checkpointing, resuming, and
the training loop itself all confirmed correct on real 22-agent data) exactly the same
way scripts/train_mappo_corridor.py was in Week 5; the actual long convergence run is
Kaggle/cloud-compute work, tracked via the same `run_name`/checkpoint mechanism so it can
be started, interrupted, and resumed across sessions without this script changing.

Usage:
    python scripts/train_mappo_district.py [--iterations 40] [--num-steps 128]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import libsumo  # noqa: F401  (must import before mlflow -- see train_mappo_corridor.py)
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atlas.common.config import project_root  # noqa: E402
from atlas.common.logging import configure_logging  # noqa: E402
from atlas.control.shield import ShieldConfig  # noqa: E402
from atlas.marl import ActorCritic, TrainingRunConfig, config_hash, run_training  # noqa: E402
from atlas.rl import MultiAgentSignalEnv, MultiAgentSignalEnvConfig  # noqa: E402
from atlas.twin import TwinEnvConfig  # noqa: E402

configure_logging()

NETWORK_DIR = project_root() / "data" / "networks" / "manhattan_midtown"
DISTRICT_TLS_IDS = [
    "42428283",
    "42428287",
    "42428297",
    "42430333",
    "42430338",
    "42430342",
    "42430344",
    "42430347",
    "42430350",
    "42430352",
    "42430356",
    "42432700",
    "42435663",
    "42435671",
    "42437363",
    "42439968",
    "42439972",
    "42439984",
    "42439990",
    "42439994",
    "5849918504",
    "GS_42439981",
]
MAX_LANES = 10  # max incoming-lane count across all 22 junctions (Week 4 benchmark)
MAX_GREEN_PHASES = 2  # every junction in this district has exactly 2 (Week 4 finding)
DECISION_INTERVAL_S = 5.0
DEMAND_PERIOD_S = 1800  # matches configs/demand/manhattan_midtown.yaml's period length
EPISODE_DECISIONS = int(DEMAND_PERIOD_S / DECISION_INTERVAL_S)
MODEL_PATH = project_root() / "data" / "models" / "mappo_district.pt"
CHECKPOINT_DIR = project_root() / "data" / "checkpoints"


def make_env() -> MultiAgentSignalEnv:
    net_file = NETWORK_DIR / "net.net.xml"
    route_file = NETWORK_DIR / "demand" / "pm_peak.rou.xml"
    vtypes_file = NETWORK_DIR / "vtypes.add.xml"
    if not net_file.is_file() or not route_file.is_file():
        raise RuntimeError(
            f"{net_file} or {route_file} missing. Run scripts/build_district_network.py "
            "and scripts/generate_demand.py first."
        )

    config = MultiAgentSignalEnvConfig(
        twin_config=TwinEnvConfig(
            name="mappo_district_train",
            net_file=net_file,
            route_file=route_file,
            additional_files=[vtypes_file] if vtypes_file.is_file() else [],
        ),
        tls_ids=DISTRICT_TLS_IDS,
        max_lanes=MAX_LANES,
        max_green_phases=MAX_GREEN_PHASES,
        shield_config=ShieldConfig(min_green_s=10.0, max_green_s=60.0),
        decision_interval_s=DECISION_INTERVAL_S,
    )
    return MultiAgentSignalEnv(config)


def train(num_iterations: int, num_steps: int, seed: int = 42) -> ActorCritic:
    run_name = f"district_{config_hash({'tls_ids': DISTRICT_TLS_IDS, 'seed': seed})}"
    config = TrainingRunConfig(
        run_name=run_name,
        num_iterations=num_iterations,
        num_steps=num_steps,
        episode_decisions=EPISODE_DECISIONS,
        seed=seed,
        checkpoint_dir=CHECKPOINT_DIR,
        mlflow_experiment="atlas-mappo-district",
    )
    net = run_training(make_env, config)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), MODEL_PATH)
    return net


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(num_iterations=args.iterations, num_steps=args.num_steps, seed=args.seed)


if __name__ == "__main__":
    main()
