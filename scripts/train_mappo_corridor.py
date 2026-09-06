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
target district before Week 6's full 25-junction run.

The rest of the district still simulates (same net/route files as the full-district
benchmark) but runs its own default fixed-time programs -- only the 5 corridor junctions
are under agent control. This matches how Week 6's curriculum actually scales: the same
env/network code, a growing `tls_ids` list.

Rollout loop: a CleanRL-style on-policy loop -- collect `num_steps` decisions (all 5
agents stepping the one shared twin in lockstep, per MultiAgentSignalEnv's design),
compute GAE, run one PPOTrainer.update(), repeat. Episodes are truncated every
`episode_decisions` decisions (the pm_peak demand period's real duration / decision
interval) since MultiAgentSignalEnv itself has no natural end condition (Week 4's
documented design) -- consistent with how train_single_junction_ppo.py treats episode
length as the training script's concern, just via manual truncation here instead of
gymnasium.wrappers.TimeLimit (dict-in/dict-out multi-agent APIs aren't TimeLimit-
compatible).

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
from typing import cast

import libsumo  # noqa: F401  (see module docstring: must import before mlflow)
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atlas.common.config import project_root  # noqa: E402
from atlas.common.logging import configure_logging, get_logger  # noqa: E402
from atlas.control.shield import ShieldConfig  # noqa: E402
from atlas.marl import ActorCritic, PPOConfig, PPOTrainer, RolloutBuffer  # noqa: E402
from atlas.rl import MultiAgentSignalEnv, MultiAgentSignalEnvConfig  # noqa: E402
from atlas.twin import TwinEnvConfig  # noqa: E402

configure_logging()
logger = get_logger(__name__)

NETWORK_DIR = project_root() / "data" / "networks" / "manhattan_midtown"
CORRIDOR_TLS_IDS = ["42430333", "42430338", "42430342", "42430344", "42430347"]
MAX_LANES = 10  # 42430333 alone has 10 incoming lanes; verified against MaxPressureProgram
MAX_GREEN_PHASES = 2  # every corridor junction has exactly 2, verified before writing this
DECISION_INTERVAL_S = 5.0
DEMAND_PERIOD_S = 1800  # matches configs/demand/manhattan_midtown.yaml's period length
EPISODE_DECISIONS = int(DEMAND_PERIOD_S / DECISION_INTERVAL_S)
MODEL_PATH = project_root() / "data" / "models" / "mappo_corridor.pt"


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


class RunningRewardNormalizer:
    """Tracks a running mean/variance of reward magnitude (Welford's online algorithm,
    the standard OpenAI-baselines RunningMeanStd formulation) and scales rewards by the
    running std before they enter the buffer. Found necessary, not applied speculatively:
    an initial training run logged value_loss ranging from ~1e3 to ~6e5 across
    iterations, traced to `total_pressure()` (atlas.control.max_pressure) being an
    unbounded raw queue-count difference -- MultiAgentSignalEnv's reward is `-pressure`
    with no normalization (Week 4, frozen; unlike the observation vector, which already
    caps queue length via `queue_norm_cap`). That gave the critic a wildly non-stationary
    regression target. This is a training-script-level fix, not a change to the frozen
    env's reward semantics: only what THIS trainer does with the reward before handing it
    to PPO changes, so evaluation code reading `infos[tls_id]["pressure"]` directly (as
    scripts/train_mappo_corridor.py's own smoke check does) is unaffected.

    Scales, not centers: dividing by the running std fixes the magnitude instability
    without shifting which rewards count as "good" vs "bad" (pressure's natural sign
    already means something -- see max_pressure.py's docstring). One shared scalar across
    all 5 agents, not per-agent: every corridor junction's pressure is the same
    queue-count-difference quantity in the same units, so a shared scale is a reasonable
    simplification, not an assumption that the junctions see identical traffic volumes.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        self.mean = 0.0
        self.var = 1.0
        self.count = eps
        self._eps = eps

    def update(self, rewards: torch.Tensor) -> None:
        batch_mean = rewards.mean().item()
        batch_var = rewards.var(unbiased=False).item()
        batch_count = rewards.numel()

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total_count

        self.mean = new_mean
        self.var = m2 / total_count
        self.count = total_count

    def scale(self, rewards: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, rewards / (self.var**0.5 + self._eps))


def _stack_obs(obs_dict: dict[str, np.ndarray], agents: list[str]) -> torch.Tensor:
    return torch.tensor(np.stack([obs_dict[a] for a in agents]), dtype=torch.float32)


def _bootstrap_values(net: ActorCritic, obs_tensor: torch.Tensor, num_agents: int) -> torch.Tensor:
    joint_obs = obs_tensor.reshape(-1).unsqueeze(0).expand(num_agents, -1)
    agent_id = torch.arange(num_agents)
    with torch.no_grad():
        return net.get_value(joint_obs, agent_id)


def train(num_iterations: int, num_steps: int, seed: int = 42) -> ActorCritic:
    torch.manual_seed(seed)
    env = make_env()
    agents = env.agents
    num_agents = len(agents)

    net = ActorCritic(env.observation_dim, env.action_dim, num_agents)
    trainer = PPOTrainer(net, PPOConfig())
    buffer = RolloutBuffer(num_steps, num_agents, env.observation_dim, env.action_dim)
    reward_normalizer = RunningRewardNormalizer()

    use_mlflow = False
    try:
        import mlflow

        use_mlflow = True
        mlflow.set_experiment("atlas-mappo-corridor")
        mlflow.start_run()
        mlflow.log_params(
            {
                "num_iterations": num_iterations,
                "num_steps": num_steps,
                "num_agents": num_agents,
                "episode_decisions": EPISODE_DECISIONS,
                **trainer.config.model_dump(),
            }
        )
    except ImportError:
        logger.warning("mlflow.not_installed", detail="metrics will be logged, not tracked")

    obs_dict = env.reset(seed=seed)
    episode_step = 0

    try:
        for iteration in range(num_iterations):
            buffer.reset()
            episode_rewards: list[float] = []

            for _ in range(num_steps):
                obs_tensor = _stack_obs(obs_dict, agents)
                joint_obs = obs_tensor.reshape(-1)
                agent_id = torch.arange(num_agents)
                with torch.no_grad():
                    actions, log_probs, _ = net.act(obs_tensor)
                    values = net.get_value(joint_obs.unsqueeze(0).expand(num_agents, -1), agent_id)

                action_dict = {a: int(actions[i].item()) for i, a in enumerate(agents)}
                next_obs_dict, reward_dict, _, _, _ = env.step(action_dict)
                rewards_tensor = torch.tensor([reward_dict[a] for a in agents], dtype=torch.float32)
                episode_rewards.append(rewards_tensor.mean().item())  # raw, for logging
                reward_normalizer.update(rewards_tensor)
                scaled_rewards = reward_normalizer.scale(rewards_tensor)

                episode_step += 1
                episode_done = episode_step >= EPISODE_DECISIONS
                buffer.insert(
                    obs=obs_tensor,
                    joint_obs=joint_obs,
                    actions=actions,
                    log_probs=log_probs,
                    rewards=scaled_rewards,
                    values=values,
                    done=episode_done,
                )

                if episode_done:
                    episode_step = 0
                    next_obs_dict = env.reset(seed=seed + iteration)
                obs_dict = next_obs_dict

            last_values = _bootstrap_values(net, _stack_obs(obs_dict, agents), num_agents)
            buffer.compute_gae(last_values=last_values, last_done=(episode_step == 0))
            stats = trainer.update(buffer)

            mean_reward = sum(episode_rewards) / len(episode_rewards)
            logger.info(
                "train.iteration",
                iteration=iteration,
                mean_reward=mean_reward,
                policy_loss=stats.policy_loss,
                value_loss=stats.value_loss,
                entropy=stats.entropy,
                clip_fraction=stats.clip_fraction,
                approx_kl=stats.approx_kl,
            )
            if use_mlflow:
                import mlflow

                mlflow.log_metrics(
                    {
                        "mean_reward": mean_reward,
                        "policy_loss": stats.policy_loss,
                        "value_loss": stats.value_loss,
                        "entropy": stats.entropy,
                        "clip_fraction": stats.clip_fraction,
                        "approx_kl": stats.approx_kl,
                    },
                    step=iteration,
                )
    finally:
        env.close()
        if use_mlflow:
            import mlflow

            mlflow.end_run()

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), MODEL_PATH)
    logger.info("train.done", model_path=str(MODEL_PATH))
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
