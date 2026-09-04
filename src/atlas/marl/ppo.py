"""The PPO update: given a full RolloutBuffer, run the clipped-surrogate objective over
multiple epochs of shuffled minibatches -- the trainer that ties together the actor-
critic network (network.py) and the buffer (buffer.py), per the frozen architecture's
"MAPPO implemented from scratch (CleanRL-style, ~500 lines)"
(docs/design/07_tech_stack.md section 7.2).

Clipping asymmetry, verified by hand before writing this (not assumed from memory -- see
tests/unit/test_marl_ppo.py): for a POSITIVE advantage (a good action) and a ratio far
outside [1-clip_coef, 1+clip_coef], the clipped objective is LESS rewarding than the
unclipped one would be -- clipping limits how much a single update can chase a good
action. For a NEGATIVE advantage (a bad action) and the same large ratio, the clipped
objective is IDENTICAL to the unclipped one -- clipping never limits the penalty for a
clearly bad update. This asymmetry is the entire point of PPO's clipped objective, and
getting it backwards is a classic, easy-to-miss implementation bug.

Flattening: RolloutBuffer stores (T, N, ...) -- T timesteps, N agents. Training samples
are (T*N,): `obs`/`actions`/`log_probs`/`advantages`/`returns` flatten directly (N varies
fastest, matching a plain `.reshape(T*N, ...)`), but `joint_obs` (T, N*obs_dim) has to be
broadcast across the agent axis first (every agent at timestep t trains against the SAME
joint observation, paired with ITS OWN agent_id) before flattening.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from pydantic import BaseModel, Field
from torch import Tensor

from atlas.marl.buffer import RolloutBuffer
from atlas.marl.network import ActorCritic


class PPOConfig(BaseModel):
    learning_rate: float = Field(default=3e-4, gt=0)
    num_epochs: int = Field(default=4, ge=1)
    minibatch_size: int = Field(default=64, ge=1)
    clip_coef: float = Field(default=0.2, gt=0)
    value_coef: float = Field(default=0.5, ge=0)
    entropy_coef: float = Field(default=0.01, ge=0)
    max_grad_norm: float = Field(default=0.5, gt=0)
    normalize_advantages: bool = True
    clip_value_loss: bool = True


@dataclass(frozen=True, slots=True)
class PPOUpdateStats:
    policy_loss: float
    value_loss: float
    entropy: float
    clip_fraction: float
    approx_kl: float


def _flatten_buffer(
    buffer: RolloutBuffer,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Returns (obs, joint_obs, agent_id, actions, log_probs, advantages, returns,
    action_masks), all flattened to (T*N, ...).
    """
    t_steps, n_agents = buffer.num_steps, buffer.num_agents

    obs = buffer.obs.reshape(t_steps * n_agents, buffer.obs_dim)
    action_masks = buffer.action_masks.reshape(t_steps * n_agents, buffer.action_dim)
    actions = buffer.actions.reshape(t_steps * n_agents)
    log_probs = buffer.log_probs.reshape(t_steps * n_agents)
    advantages = buffer.advantages.reshape(t_steps * n_agents)
    returns = buffer.returns.reshape(t_steps * n_agents)

    joint_obs = (
        buffer.joint_obs.unsqueeze(1)
        .expand(t_steps, n_agents, n_agents * buffer.obs_dim)
        .reshape(t_steps * n_agents, n_agents * buffer.obs_dim)
    )
    agent_id = torch.arange(n_agents).repeat(t_steps)

    return obs, joint_obs, agent_id, actions, log_probs, advantages, returns, action_masks


def clipped_policy_loss(ratio: Tensor, advantages: Tensor, clip_coef: float) -> Tensor:
    """PPO's clipped surrogate objective (negated to a loss to minimize). Asymmetric by
    design, verified by hand before this was written (see the module docstring and
    tests/unit/test_marl_ppo.py): clipping limits how much a large ratio can inflate the
    objective for a GOOD action (positive advantage), but never limits the penalty for a
    clearly BAD action (negative advantage) -- that asymmetry is the entire point of PPO.
    """
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef) * advantages
    return -torch.min(surr1, surr2).mean()


def clipped_value_loss(
    new_values: Tensor, old_values: Tensor, returns: Tensor, clip_coef: float
) -> Tensor:
    """PPO2-style value clipping: the value function isn't allowed to move further than
    `clip_coef` from its old estimate in one update, same trust-region motivation as the
    policy clip. Takes the worse (larger) of the clipped/unclipped squared error, so
    clipping can only ever reduce the incentive to change the value estimate a lot, never
    increase it.
    """
    clipped_values = old_values + torch.clamp(new_values - old_values, -clip_coef, clip_coef)
    value_loss_unclipped = (new_values - returns) ** 2
    value_loss_clipped = (clipped_values - returns) ** 2
    return 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()


class PPOTrainer:
    def __init__(self, network: ActorCritic, config: PPOConfig | None = None) -> None:
        self.network = network
        self.config = config or PPOConfig()
        self.optimizer = torch.optim.Adam(network.parameters(), lr=self.config.learning_rate)

    def update(self, buffer: RolloutBuffer) -> PPOUpdateStats:
        if not buffer.full:
            raise RuntimeError(
                "Buffer must be full (and have compute_gae() called) before update()"
            )

        obs, joint_obs, agent_id, actions, old_log_probs, advantages, returns, action_masks = (
            _flatten_buffer(buffer)
        )
        batch_size = obs.shape[0]
        minibatch_size = min(self.config.minibatch_size, batch_size)

        policy_losses, value_losses, entropies, clip_fractions, approx_kls = [], [], [], [], []

        for _ in range(self.config.num_epochs):
            permutation = torch.randperm(batch_size)
            for start in range(0, batch_size, minibatch_size):
                idx = permutation[start : start + minibatch_size]

                mb_advantages = advantages[idx]
                if self.config.normalize_advantages and mb_advantages.numel() > 1:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                new_log_probs, entropy, new_values = self.network.evaluate_actions(
                    obs[idx], joint_obs[idx], agent_id[idx], actions[idx], action_masks[idx]
                )

                log_ratio = new_log_probs - old_log_probs[idx]
                ratio = log_ratio.exp()

                policy_loss = clipped_policy_loss(ratio, mb_advantages, self.config.clip_coef)

                mb_returns = returns[idx]
                if self.config.clip_value_loss:
                    old_values = buffer.values.reshape(-1)[idx]
                    value_loss = clipped_value_loss(
                        new_values, old_values, mb_returns, self.config.clip_coef
                    )
                else:
                    value_loss = 0.5 * ((new_values - mb_returns) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()  # type: ignore[no-untyped-call]
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    clip_fraction = ((ratio - 1.0).abs() > self.config.clip_coef).float().mean()
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy_loss.item())
                clip_fractions.append(clip_fraction.item())
                approx_kls.append(approx_kl.item())

        return PPOUpdateStats(
            policy_loss=sum(policy_losses) / len(policy_losses),
            value_loss=sum(value_losses) / len(value_losses),
            entropy=sum(entropies) / len(entropies),
            clip_fraction=sum(clip_fractions) / len(clip_fractions),
            approx_kl=sum(approx_kls) / len(approx_kls),
        )
