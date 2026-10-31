"""Generic MAPPO training loop (CleanRL-style on-policy rollout + PPO update), shared by
every curriculum-stage training script (corridor -- Week 5, district -- Week 6, ...) so
the loop is written once. Extracted here once a second training script (district-scale)
needed the exact same loop `scripts/train_mappo_corridor.py` (Week 5) already had, rather
than duplicating it -- not built speculatively ahead of a second caller.

Checkpointing (docs/design/06_ai_pipeline.md section 6.1's "checkpoint every 10 min...
fully resumable, session death is normal, not exceptional"): every `checkpoint_every_s`
WALL-CLOCK seconds (default 600s, matching that literal "10 min", not an iteration count
-- iteration wall-clock time varies by hardware/env size, and Kaggle-session-death safety
depends on real elapsed time), the trainer's full state -- network weights, optimizer
state, AND the reward normalizer's running statistics (all three are needed to resume
training identically, not just the weights, since Week 5's `RunningRewardNormalizer`
scales every reward the optimizer ever saw) -- is written to one `<run_name>_latest.pt`
file, overwritten each time. `run_name` should be config-hashed by the caller (see
`config_hash()`) so a genuinely different run doesn't silently resume from an unrelated
one's checkpoint.

Import-order note: this module deliberately does NOT import mlflow or libsumo itself --
both stay the caller's concern (see scripts/train_mappo_corridor.py's module docstring
for the real DLL-collision reason libsumo must be imported before mlflow in any script
using both; importing either here would force every caller into that ordering
constraint even if they don't need mlflow tracking).
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field

from atlas.common.logging import get_logger
from atlas.marl.buffer import RolloutBuffer
from atlas.marl.network import ActorCritic
from atlas.marl.ppo import PPOConfig, PPOTrainer
from atlas.rl.multi_agent_env import MultiAgentSignalEnv

logger = get_logger(__name__)


class RunningRewardNormalizer:
    """Tracks a running mean/variance of reward magnitude (Welford's online algorithm,
    the standard OpenAI-baselines RunningMeanStd formulation) and scales rewards by the
    running std before they enter the buffer. Found necessary during Week 5's corridor
    run, not applied speculatively: an initial training run logged value_loss ranging
    from ~1e3 to ~6e5 across iterations, traced to `total_pressure()`
    (atlas.control.max_pressure) being an unbounded raw queue-count difference --
    `MultiAgentSignalEnv`'s reward is `-pressure` with no normalization (Week 4, frozen;
    unlike the observation vector, which already caps queue length via
    `queue_norm_cap`). That gave the critic a wildly non-stationary regression target.
    This is a training-loop-level fix, not a change to the frozen env's reward
    semantics: only what the trainer does with the reward before handing it to PPO
    changes, so evaluation code reading `infos[tls_id]["pressure"]` directly is
    unaffected.

    Scales, not centers: dividing by the running std fixes the magnitude instability
    without shifting which rewards count as "good" vs "bad" (pressure's natural sign
    already means something -- see max_pressure.py's docstring). One shared scalar
    across all agents, not per-agent: every junction's pressure is the same
    queue-count-difference quantity in the same units, so a shared scale is a
    reasonable simplification, not an assumption that every junction sees identical
    traffic volumes.
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

    def state_dict(self) -> dict[str, float]:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state: dict[str, float]) -> None:
        self.mean = state["mean"]
        self.var = state["var"]
        self.count = state["count"]


class TrainingRunConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    run_name: str
    num_iterations: int = Field(gt=0)
    num_steps: int = Field(gt=0)
    episode_decisions: int = Field(gt=0)
    seed: int = 42
    checkpoint_dir: Path
    checkpoint_every_s: float = Field(default=600.0, gt=0)
    mlflow_experiment: str | None = None
    max_wall_s: float | None = Field(default=None, gt=0)
    """Stop after the first iteration that finishes past this many wall-clock seconds,
    saving a resumable checkpoint. For time-boxed sessions (a Kaggle/Colab run has a hard
    session limit): ask for far more iterations than fit and let the budget end the run
    cleanly, then resume from the checkpoint in the next session."""
    ppo_config: PPOConfig = Field(default_factory=PPOConfig)


def config_hash(payload: dict[str, Any]) -> str:
    """A short, deterministic hash of a run's hyperparameters/env identity, so two
    genuinely different configurations never collide on the same checkpoint file, and
    re-running the SAME configuration resumes from its own checkpoint automatically.
    """
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]


def checkpoint_path(checkpoint_dir: Path, run_name: str) -> Path:
    return checkpoint_dir / f"{run_name}_latest.pt"


def save_checkpoint(
    path: Path,
    iteration: int,
    net: ActorCritic,
    optimizer: torch.optim.Optimizer,
    reward_normalizer: RunningRewardNormalizer,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "iteration": iteration,
            "net_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "reward_normalizer": reward_normalizer.state_dict(),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    net: ActorCritic,
    optimizer: torch.optim.Optimizer,
    reward_normalizer: RunningRewardNormalizer,
) -> int:
    """Loads network/optimizer/normalizer state in place and returns the last
    COMPLETED iteration index -- the caller resumes at `return_value + 1`.
    """
    checkpoint = torch.load(path, weights_only=True)
    net.load_state_dict(checkpoint["net_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    reward_normalizer.load_state_dict(checkpoint["reward_normalizer"])
    return cast(int, checkpoint["iteration"])


def _stack_obs(obs_dict: dict[str, np.ndarray], agents: list[str]) -> torch.Tensor:
    return torch.tensor(np.stack([obs_dict[a] for a in agents]), dtype=torch.float32)


def _bootstrap_values(net: ActorCritic, obs_tensor: torch.Tensor, num_agents: int) -> torch.Tensor:
    joint_obs = obs_tensor.reshape(-1).unsqueeze(0).expand(num_agents, -1)
    agent_id = torch.arange(num_agents)
    with torch.no_grad():
        return net.get_value(joint_obs, agent_id)


def run_training(
    env_factory: Callable[[], MultiAgentSignalEnv],
    config: TrainingRunConfig,
) -> ActorCritic:
    """The CleanRL-style loop: collect `config.num_steps` decisions, compute GAE, run
    one `PPOTrainer.update()`, repeat for `config.num_iterations` -- resuming from
    `config.checkpoint_dir`'s checkpoint for `config.run_name` if one exists. Episodes
    are truncated every `config.episode_decisions` decisions (`MultiAgentSignalEnv` has
    no natural end condition -- Week 4's documented design), matching Week 5's manual-
    truncation approach.
    """
    torch.manual_seed(config.seed)
    env = env_factory()
    agents = env.agents
    num_agents = len(agents)

    net = ActorCritic(env.observation_dim, env.action_dim, num_agents)
    trainer = PPOTrainer(net, config.ppo_config)
    buffer = RolloutBuffer(config.num_steps, num_agents, env.observation_dim, env.action_dim)
    reward_normalizer = RunningRewardNormalizer()

    ckpt_path = checkpoint_path(config.checkpoint_dir, config.run_name)
    start_iteration = 0
    if ckpt_path.is_file():
        last_completed = load_checkpoint(ckpt_path, net, trainer.optimizer, reward_normalizer)
        start_iteration = last_completed + 1
        logger.info("train.resumed", checkpoint=str(ckpt_path), from_iteration=start_iteration)

    use_mlflow = False
    if config.mlflow_experiment is not None:
        try:
            import mlflow

            use_mlflow = True
            mlflow.set_experiment(config.mlflow_experiment)
            mlflow.start_run(run_name=config.run_name)
            mlflow.log_params(
                {
                    "run_name": config.run_name,
                    "num_iterations": config.num_iterations,
                    "num_steps": config.num_steps,
                    "num_agents": num_agents,
                    "episode_decisions": config.episode_decisions,
                    "resumed_from_iteration": start_iteration if start_iteration > 0 else None,
                    **config.ppo_config.model_dump(),
                }
            )
        except ImportError:
            logger.warning("mlflow.not_installed", detail="metrics will be logged, not tracked")

    obs_dict = env.reset(seed=config.seed)
    run_started = time.monotonic()
    episode_step = 0
    last_checkpoint_time = time.monotonic()
    iteration = start_iteration - 1

    try:
        for iteration in range(start_iteration, config.num_iterations):
            buffer.reset()
            episode_rewards: list[float] = []

            for _ in range(config.num_steps):
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
                episode_done = episode_step >= config.episode_decisions
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
                    next_obs_dict = env.reset(seed=config.seed + iteration)
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

            if time.monotonic() - last_checkpoint_time >= config.checkpoint_every_s:
                save_checkpoint(ckpt_path, iteration, net, trainer.optimizer, reward_normalizer)
                last_checkpoint_time = time.monotonic()
                logger.info("train.checkpoint", iteration=iteration, path=str(ckpt_path))

            if (
                config.max_wall_s is not None
                and time.monotonic() - run_started >= config.max_wall_s
            ):
                logger.info(
                    "train.time_budget_reached",
                    iteration=iteration,
                    budget_s=config.max_wall_s,
                    resume_with="the same command -- it continues from the saved checkpoint",
                )
                break
    finally:
        if iteration >= 0:
            save_checkpoint(ckpt_path, iteration, net, trainer.optimizer, reward_normalizer)
        env.close()
        if use_mlflow:
            import mlflow

            mlflow.end_run()

    return net
