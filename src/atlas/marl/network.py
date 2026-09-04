"""The MAPPO actor-critic network: shared-parameter actor + centralized critic (CTDE --
centralized training, decentralized execution), from scratch in PyTorch, per the frozen
architecture's explicit decision (docs/design/07_tech_stack.md section 7.2: "write MAPPO
ourselves... learning value, debuggability, interview depth" over RLlib) and
docs/design/06_ai_pipeline.md section 6.1.

Parameter sharing: ONE actor network serves every junction in a training run (the
scaling recipe this project's control plane is built on, matching MPLight/PressLight --
see docs/design/02_literature_survey.md section 2.4). Its forward pass takes exactly one
agent's local observation (the padded, fixed-size vector `MultiAgentSignalEnv` already
produces -- Week 4), so the same network runs for every agent regardless of district
size.

Centralized critic: sees the CONCATENATED observations of every agent in the training
run, used only during training to compute the value baseline for advantage estimation --
never during execution (Part 6.1's "centralized critic sees the concatenated district
state during training only"). Its input dimension is therefore fixed to one specific
`num_agents` (one curriculum stage = one critic instance, not something reused across
different district sizes).

Action masking: an agent whose real green-phase count is less than the padded
`max_green_phases` (Week 4's MultiAgentSignalEnv again) must never have probability mass
on the nonexistent padding actions -- masked logits are set to -inf before the softmax,
a standard, correct pattern (verified here, not just assumed, against a real degenerate
case: an agent with only 1 legal action out of a padded 3).
"""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn
from torch.distributions import Categorical


def build_action_mask(num_legal_actions: int, max_actions: int) -> Tensor:
    """A boolean mask of shape (max_actions,): True for the first `num_legal_actions`
    slots, False for padding. `num_legal_actions` is a junction's real green-phase count
    (from MaxPressureProgram.green_phase_indices); `max_actions` is the training run's
    padded `max_green_phases` (Week 4).
    """
    if num_legal_actions < 1:
        raise ValueError(f"num_legal_actions must be >= 1, got {num_legal_actions}")
    if num_legal_actions > max_actions:
        raise ValueError(
            f"num_legal_actions ({num_legal_actions}) must be <= max_actions ({max_actions})"
        )
    mask = torch.zeros(max_actions, dtype=torch.bool)
    mask[:num_legal_actions] = True
    return mask


class ActorCritic(nn.Module):
    def __init__(
        self, obs_dim: int, action_dim: int, num_agents: int, hidden_dim: int = 256
    ) -> None:
        super().__init__()
        if obs_dim < 1 or action_dim < 1 or num_agents < 1:
            raise ValueError("obs_dim, action_dim, and num_agents must all be >= 1")
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents

        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim * num_agents, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def action_logits(self, obs: Tensor, action_mask: Tensor | None = None) -> Tensor:
        """obs: (batch, obs_dim). action_mask: (batch, action_dim) bool, True = legal.
        Masked-out logits go to -inf before the caller builds a Categorical from this, so
        the softmax assigns them exactly zero probability -- never partial, never NaN, as
        long as every row has at least one True (build_action_mask enforces this at
        construction).
        """
        logits = self.actor(obs)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, float("-inf"))
        return cast(Tensor, logits)

    def act(
        self, obs: Tensor, action_mask: Tensor | None = None, deterministic: bool = False
    ) -> tuple[Tensor, Tensor, Tensor]:
        """For execution (decentralized -- no critic involved): sample (or argmax) an
        action from one agent's local observation. Returns (action, log_prob, entropy).
        """
        logits = self.action_logits(obs, action_mask)
        dist = Categorical(logits=logits)
        action = (
            torch.argmax(logits, dim=-1) if deterministic else dist.sample()  # type: ignore[no-untyped-call]
        )
        log_prob = dist.log_prob(action)  # type: ignore[no-untyped-call]
        entropy = dist.entropy()  # type: ignore[no-untyped-call]
        return action, log_prob, entropy

    def evaluate_actions(
        self, obs: Tensor, joint_obs: Tensor, action: Tensor, action_mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """For training (centralized -- the critic sees joint_obs): the log-probability
        and entropy of previously-taken `action` under the current policy, plus the
        critic's value estimate from the joint observation. Returns
        (log_prob, entropy, value).
        """
        logits = self.action_logits(obs, action_mask)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(action)  # type: ignore[no-untyped-call]
        entropy = dist.entropy()  # type: ignore[no-untyped-call]
        value = cast(Tensor, self.critic(joint_obs).squeeze(-1))
        return log_prob, entropy, value

    def get_value(self, joint_obs: Tensor) -> Tensor:
        return cast(Tensor, self.critic(joint_obs).squeeze(-1))
