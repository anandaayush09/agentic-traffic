#!/usr/bin/env python
"""Trains the from-scratch MAPPO trainer (atlas.marl) on manhattan_midtown's corridor
curriculum stage -- the Week 5 roadmap deliverable (docs/design/13_roadmap.md, Week 5:
"MAPPO trains on the corridor curriculum stage (5 junctions), tracked in MLflow"), the
first time network.py/buffer.py/ppo.py run together against a real simulation rather than
synthetic tensors.

Corridor selection: manhattan_midtown has 22 signalized junctions from a 6-street x
3-avenue grid (Week 2's AUDIT.md). Checked every junction's coordinates directly
(not assumed) via `sumolib`: 7 of them --
42430333, 42430338, 42430342, 42430344, 42430347, 42430350, 42430356 -- fall on one
almost-perfectly-collinear line (~38, ~70 constant x/y delta between consecutive points),
i.e. one avenue crossing 7 consecutive streets. The first 5 of those are the "corridor":
a real, spatially contiguous sub-network, not an arbitrary subset -- the standard
curriculum-learning recipe of training parameter-shared MAPPO on a small slice of the
target district before Week 6's full-district run.

The rest of the district still simulates (same net/route files as the full-district
benchmark) but runs its own default fixed-time programs -- only the 5 corridor junctions
are under agent control. This matches how Week 6's curriculum actually scales: the same
env/network code, a growing `tls_ids` list -- see scripts/train_mappo_district.py.

Training loop itself (rollout collection, GAE, PPO update, checkpointing) lives in
atlas.marl.training.run_training -- extracted there in Week 6 once train_mappo_district.py
needed the exact same loop, rather than duplicated across two scripts.

Action masking is a documented no-op here: all 5 corridor junctions have exactly 2 green
phases (verified against max_green_phases before writing this), matching Week 4's own
"moot for both networks today" note -- RolloutBuffer.insert()'s default all-legal mask is
used throughout.

Import-order gotcha (found running this script, not assumed): `import mlflow` before
`import libsumo` in the same process breaks libsumo's native DLL loading on Windows
(`ImportError: DLL load failed while importing _libsumo: The specified procedure could
not be found`) -- a real conflict, reproduced in isolation with a bare `import mlflow;
import libsumo`, and NOT order-independent: `import libsumo; import mlflow` works fine.
Root cause: libsumo bundles its own Arrow-linked native library and mlflow transitively
pulls in a different pyarrow build (libsumo's own startup warning names this exact
mismatch); whichever loads its Arrow symbols into the process first wins. Same class of
bug as atlas/rl/rollout.py's documented Ray/libsumo DLL collision, different culprit
library. Fixed the same way: import libsumo eagerly, before mlflow gets a chance to.

Usage:
    python scripts/train_mappo_corridor.py [--iterations 40] [--num-steps 128]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import libsumo  # noqa: F401  (see module docstring: must import before mlflow)
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
CORRIDOR_TLS_IDS = ["42430333", "42430338", "42430342", "42430344", "42430347"]
MAX_LANES = 10  # 42430333 alone has 10 incoming lanes; verified against MaxPressureProgram
MAX_GREEN_PHASES = 2  # every corridor junction has exactly 2, verified before writing this
DECISION_INTERVAL_S = 5.0
DEMAND_PERIOD_S = 1800  # matches configs/demand/manhattan_midtown.yaml's period length
EPISODE_DECISIONS = int(DEMAND_PERIOD_S / DECISION_INTERVAL_S)
MODEL_PATH = project_root() / "data" / "models" / "mappo_corridor.pt"
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
            name="mappo_corridor_train",
            net_file=net_file,
            route_file=route_file,
            additional_files=[vtypes_file] if vtypes_file.is_file() else [],
        ),
        tls_ids=CORRIDOR_TLS_IDS,
        max_lanes=MAX_LANES,
        max_green_phases=MAX_GREEN_PHASES,
        shield_config=ShieldConfig(min_green_s=10.0, max_green_s=60.0),
        decision_interval_s=DECISION_INTERVAL_S,
    )
    return MultiAgentSignalEnv(config)


def train(num_iterations: int, num_steps: int, seed: int = 42) -> ActorCritic:
    run_name = f"corridor_{config_hash({'tls_ids': CORRIDOR_TLS_IDS, 'seed': seed})}"
    config = TrainingRunConfig(
        run_name=run_name,
        num_iterations=num_iterations,
        num_steps=num_steps,
        episode_decisions=EPISODE_DECISIONS,
        seed=seed,
        checkpoint_dir=CHECKPOINT_DIR,
        mlflow_experiment="atlas-mappo-corridor",
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
