#!/usr/bin/env python
"""Evaluates the trained PPO policy against two classical baselines (max-pressure,
native fixed-time) on held-out evaluation demand (a different seed than training used --
see scripts/generate_rl_demand.py) -- the Week 3 roadmap's explicit deliverable: "SB3
single-junction agent beats fixed-time on a toy network."

All three controllers run through the identical SingleJunctionEnv.step() path (same
shield, same reward/pressure accounting), so the comparison isolates the actual control
decisions rather than any difference in how each was measured.

Usage:
    python scripts/evaluate_single_junction.py
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atlas.common.config import project_root  # noqa: E402
from atlas.common.logging import configure_logging, get_logger  # noqa: E402
from atlas.control.fixed_time import (  # noqa: E402
    native_green_durations_from_net_file,
    select_fixed_time_phase,
)
from atlas.control.max_pressure import MaxPressureProgram, select_max_pressure_phase  # noqa: E402
from atlas.control.shield import ShieldConfig  # noqa: E402
from atlas.rl import SingleJunctionEnv, SingleJunctionEnvConfig  # noqa: E402
from atlas.twin import TwinEnvConfig  # noqa: E402

configure_logging()
logger = get_logger(__name__)

NETWORK_DIR = project_root() / "data" / "networks" / "sandbox"
NET_FILE = NETWORK_DIR / "net.net.xml"
TLS_ID = "B1"
DECISION_INTERVAL_S = 5.0
EPISODE_DURATION_S = 1200  # matches scripts/generate_rl_demand.py
MODEL_PATH = project_root() / "data" / "models" / "single_junction_ppo.zip"

Policy = Callable[[np.ndarray, SingleJunctionEnv], int]


def make_env(route_file: Path) -> SingleJunctionEnv:
    config = SingleJunctionEnvConfig(
        twin_config=TwinEnvConfig(name="rl_eval", net_file=NET_FILE, route_file=route_file),
        tls_id=TLS_ID,
        shield_config=ShieldConfig(min_green_s=10.0, max_green_s=60.0),
        decision_interval_s=DECISION_INTERVAL_S,
    )
    return SingleJunctionEnv(config)


def run_episode(env: SingleJunctionEnv, policy: Policy) -> dict[str, float]:
    obs, _ = env.reset()
    max_steps = int(EPISODE_DURATION_S / DECISION_INTERVAL_S)
    rewards: list[float] = []
    pressures: list[float] = []
    switches = 0
    for _ in range(max_steps):
        action = policy(obs, env)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        pressures.append(info["pressure"])
        if info["phase_changed"]:
            switches += 1
        if terminated or truncated:
            break
    return {
        "mean_reward": sum(rewards) / len(rewards),
        "mean_pressure": sum(pressures) / len(pressures),
        "total_switches": float(switches),
        "steps": float(len(rewards)),
    }


def make_ppo_policy(model_path: Path) -> Policy:
    from stable_baselines3 import PPO

    from atlas.rl import patch_sb3_zip_loading

    # Windows workaround for a real SB3/torch incompatibility, not our own bug -- see
    # src/atlas/rl/model_io.py's module docstring for the full diagnosis.
    patch_sb3_zip_loading()
    model = PPO.load(str(model_path))

    def policy(obs: np.ndarray, env: SingleJunctionEnv) -> int:
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    return policy


def make_max_pressure_policy(mp_program: MaxPressureProgram) -> Policy:
    lane_ids = sorted(
        {m.from_lane for m in mp_program.movements} | {m.to_lane for m in mp_program.movements}
    )

    def policy(obs: np.ndarray, env: SingleJunctionEnv) -> int:
        conn = env.connection
        queue_lengths = {
            lane_id: int(conn.lane.getLastStepHaltingNumber(lane_id)) for lane_id in lane_ids
        }
        phase = select_max_pressure_phase(mp_program, queue_lengths)
        return mp_program.green_phase_indices.index(phase)

    return policy


def make_fixed_time_policy(
    green_phase_indices: tuple[int, ...], durations_s: tuple[float, ...]
) -> Policy:
    state = {"cycle_elapsed_s": -DECISION_INTERVAL_S}  # first call lands on t=0

    def policy(obs: np.ndarray, env: SingleJunctionEnv) -> int:
        state["cycle_elapsed_s"] += DECISION_INTERVAL_S
        phase = select_fixed_time_phase(green_phase_indices, durations_s, state["cycle_elapsed_s"])
        return green_phase_indices.index(phase)

    return policy


def main() -> None:
    import generate_rl_demand

    if not MODEL_PATH.is_file():
        raise RuntimeError(
            f"{MODEL_PATH} not found. Run scripts/train_single_junction_ppo.py first."
        )

    _, eval_route = generate_rl_demand.generate_all()
    mp_program = MaxPressureProgram.from_net_file(TLS_ID, NET_FILE)
    green_phases, native_durations = native_green_durations_from_net_file(TLS_ID, NET_FILE)

    policies: dict[str, Policy] = {
        "ppo": make_ppo_policy(MODEL_PATH),
        "max_pressure": make_max_pressure_policy(mp_program),
        "fixed_time": make_fixed_time_policy(green_phases, native_durations),
    }

    results: dict[str, dict[str, float]] = {}
    for name, policy in policies.items():
        env = make_env(eval_route)
        try:
            results[name] = run_episode(env, policy)
        finally:
            env.close()
        logger.info("evaluate.controller_done", controller=name, **results[name])

    print("\n=== Single-junction evaluation (B1, held-out demand) ===")
    print(f"{'controller':<14}{'mean_reward':>14}{'mean_pressure':>16}{'switches':>10}")
    for name, r in results.items():
        print(
            f"{name:<14}{r['mean_reward']:>14.2f}{r['mean_pressure']:>16.2f}{int(r['total_switches']):>10}"
        )

    ppo_reward = results["ppo"]["mean_reward"]
    fixed_time_reward = results["fixed_time"]["mean_reward"]
    verdict = "beats" if ppo_reward > fixed_time_reward else "does NOT beat"
    print(
        f"\nPPO {verdict} fixed-time on mean reward ({ppo_reward:.2f} vs {fixed_time_reward:.2f})."
    )


if __name__ == "__main__":
    main()
