"""RolloutBuffer: pre-allocated, CleanRL-style storage for one rollout (T timesteps x N
agents), plus Generalized Advantage Estimation (GAE) -- the data structure the PPO update
loop consumes, per the frozen architecture's "MAPPO implemented from scratch (CleanRL-
style, ~500 lines)" (docs/design/07_tech_stack.md section 7.2).

Shapes follow CleanRL's own convention (pre-allocate everything up front, fill by index
during collection, rather than a dynamic append-based buffer). `joint_obs` and `dones`
are stored once per timestep (T, ...), not duplicated per agent -- they describe the one
shared simulation state and episode-level termination, not anything agent-specific.

GAE is computed independently per agent-column: each agent has its own reward sequence
AND (since the per-agent-conditioned critic fix) its own value sequence, so the standard
single-agent GAE recursion just runs once per column with no cross-agent interaction --
the standard SB3/CleanRL formula, verified here against a hand-computed example rather
than assumed correct from memory (see tests/unit/test_marl_buffer.py).
"""

from __future__ import annotations

import torch
from torch import Tensor


class RolloutBuffer:
    def __init__(self, num_steps: int, num_agents: int, obs_dim: int, action_dim: int) -> None:
        if num_steps < 1 or num_agents < 1 or obs_dim < 1 or action_dim < 1:
            raise ValueError("num_steps, num_agents, obs_dim, and action_dim must all be >= 1")
        self.num_steps = num_steps
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.obs = torch.zeros(num_steps, num_agents, obs_dim)
        self.joint_obs = torch.zeros(num_steps, num_agents * obs_dim)
        self.actions = torch.zeros(num_steps, num_agents, dtype=torch.long)
        self.log_probs = torch.zeros(num_steps, num_agents)
        self.rewards = torch.zeros(num_steps, num_agents)
        self.values = torch.zeros(num_steps, num_agents)
        self.dones = torch.zeros(num_steps)
        self.action_masks = torch.ones(num_steps, num_agents, action_dim, dtype=torch.bool)

        self.advantages = torch.zeros(num_steps, num_agents)
        self.returns = torch.zeros(num_steps, num_agents)

        self._ptr = 0

    @property
    def full(self) -> bool:
        return self._ptr >= self.num_steps

    def insert(
        self,
        obs: Tensor,
        joint_obs: Tensor,
        actions: Tensor,
        log_probs: Tensor,
        rewards: Tensor,
        values: Tensor,
        done: bool,
        action_masks: Tensor | None = None,
    ) -> None:
        """obs/actions/log_probs/rewards/values: (num_agents, ...) for one timestep.
        joint_obs: (num_agents * obs_dim,). done: whether this timestep ended the
        episode. action_masks: (num_agents, action_dim), defaults to all-legal if
        omitted (matches an environment where every agent's action space is fully
        populated, no padding).
        """
        if self.full:
            raise RuntimeError(
                f"Buffer is full ({self.num_steps} steps already inserted); call reset() "
                "before inserting more."
            )
        t = self._ptr
        self.obs[t] = obs
        self.joint_obs[t] = joint_obs
        self.actions[t] = actions
        self.log_probs[t] = log_probs
        self.rewards[t] = rewards
        self.values[t] = values
        self.dones[t] = float(done)
        if action_masks is not None:
            self.action_masks[t] = action_masks
        self._ptr += 1

    def reset(self) -> None:
        self._ptr = 0

    def compute_gae(
        self, last_values: Tensor, last_done: bool, gamma: float = 0.99, gae_lambda: float = 0.95
    ) -> None:
        """Fills self.advantages and self.returns in place, per agent. `last_values`:
        (num_agents,), the critic's bootstrap value for the state immediately after the
        last stored step. `last_done`: whether that state is terminal (no bootstrap
        needed past it).
        """
        if not self.full:
            raise RuntimeError(
                f"Buffer is not full ({self._ptr}/{self.num_steps} steps inserted); GAE "
                "needs a complete rollout."
            )
        last_gae_lam = torch.zeros(self.num_agents)
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_values = self.values[t + 1]
            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[t] = last_gae_lam
        self.returns = self.advantages + self.values
