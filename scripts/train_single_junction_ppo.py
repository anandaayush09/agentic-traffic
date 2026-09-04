#!/usr/bin/env python
"""Trains a Stable-Baselines3 PPO agent on SingleJunctionEnv -- the Week 3 RL ramp-up
deliverable (docs/design/13_roadmap.md, Week 3). Saves the trained model for
scripts/evaluate_single_junction.py to compare against the classical baselines already
built (max-pressure, fixed-time).

Episodes are capped via gymnasium.wrappers.TimeLimit: SingleJunctionEnv itself has no
natural end condition (see its step() docstring), so episode length is entirely this
script's concern, not the environment's.

Usage:
    python scripts/train_single_junction_ppo.py [--timesteps 50000]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atlas.common.config import project_root  # noqa: E402
from atlas.common.logging import configure_logging, get_logger  # noqa: E402
from atlas.control.shield import ShieldConfig  # noqa: E402
from atlas.rl import SingleJunctionEnv, SingleJunctionEnvConfig  # noqa: E402
from atlas.twin import TwinEnvConfig  # noqa: E402

configure_logging()
logger = get_logger(__name__)

NETWORK_DIR = project_root() / "data" / "networks" / "sandbox"
TLS_ID = "B1"
DECISION_INTERVAL_S = 5.0
EPISODE_DURATION_S = 1200  # matches scripts/generate_rl_demand.py's DURATION_S
MODEL_PATH = project_root() / "data" / "models" / "single_junction_ppo.zip"


def make_env(route_file: Path) -> object:
    from gymnasium.wrappers import TimeLimit

    config = SingleJunctionEnvConfig(
        twin_config=TwinEnvConfig(
            name="rl_train", net_file=NETWORK_DIR / "net.net.xml", route_file=route_file
        ),
        tls_id=TLS_ID,
        shield_config=ShieldConfig(min_green_s=10.0, max_green_s=60.0),
        decision_interval_s=DECISION_INTERVAL_S,
    )
    max_episode_steps = int(EPISODE_DURATION_S / DECISION_INTERVAL_S)
    return TimeLimit(SingleJunctionEnv(config), max_episode_steps=max_episode_steps)


def train(total_timesteps: int, force_demand: bool = False) -> Path:
    from stable_baselines3 import PPO

    import generate_rl_demand

    train_route, _ = generate_rl_demand.generate_all(force=force_demand)

    env = make_env(train_route)
    model = PPO("MlpPolicy", env, verbose=1, seed=42, n_steps=512, batch_size=64)

    logger.info("train.starting", total_timesteps=total_timesteps)
    model.learn(total_timesteps=total_timesteps)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_PATH))
    logger.info("train.done", model_path=str(MODEL_PATH))

    env.close()
    return MODEL_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--force-demand", action="store_true")
    args = parser.parse_args()
    train(total_timesteps=args.timesteps, force_demand=args.force_demand)


if __name__ == "__main__":
    main()
