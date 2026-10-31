"""Loading a trained MAPPO policy for *execution* (the live twin, the evaluation harness).

Training produces two kinds of file, and execution should accept either:

- a resumable checkpoint (`<run>_latest.pt`): a dict with `net_state_dict`, the optimizer,
  the reward normalizer and the last completed iteration;
- a plain `state_dict` (what `scripts/train_mappo_*.py` write to `data/models/`).

Why a policy trained on N junctions can drive a different number of them: the *actor* is
one shared, decentralized network over a single junction's observation (a fixed-width
vector, padded to `max_lanes`), so it is indifferent to how many agents trained alongside
it. Only the *critic* depends on the agent count (it sees the joint observation), and the
critic is not used to act. So a policy trained on the full 22-junction district can run the
5-junction corridor in the evaluation, or any subset live. What it cannot do is run on an
environment whose observation or action width differs (`max_lanes` / `max_green_phases`):
`check_policy_fits` makes that mismatch a clear error instead of a shape crash mid-run.

Honest limit: loading says nothing about quality. A checkpoint from a short smoke run loads
and acts exactly like a converged one; whether it is any *good* is what the evaluation
(`scripts/run_evaluation_pilot.py --rl-checkpoint`) is for.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from atlas.marl.network import ActorCritic


@dataclass(frozen=True, slots=True)
class LoadedPolicy:
    net: ActorCritic
    source: Path
    iteration: int | None
    """Last completed training iteration, when the file is a resumable checkpoint."""
    obs_dim: int
    action_dim: int
    trained_agents: int
    """How many junctions the critic was trained over (informational: acting ignores it)."""

    @property
    def description(self) -> str:
        trained = f"iteration {self.iteration}" if self.iteration is not None else "final weights"
        return f"{self.source.name} ({trained}; trained over {self.trained_agents} junctions)"


def load_policy(path: Path) -> LoadedPolicy:
    """Loads `path` (a checkpoint or a plain state_dict) into an eval-mode `ActorCritic`
    whose dimensions are inferred from the weights."""
    if not path.is_file():
        raise FileNotFoundError(f"No policy file at {path}")
    try:
        raw: Any = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"{path.name} is not a loadable PyTorch file: {exc}") from exc

    iteration: int | None = None
    if isinstance(raw, dict) and "net_state_dict" in raw:
        state = raw["net_state_dict"]
        iteration = int(raw["iteration"]) if "iteration" in raw else None
    else:
        state = raw
    if not isinstance(state, dict):
        raise ValueError(f"{path.name} does not contain a state_dict")

    try:
        hidden_dim, obs_dim = state["actor.0.weight"].shape
        action_dim = int(state["actor.4.weight"].shape[0])
        critic_in = int(state["critic.0.weight"].shape[1])
    except (KeyError, AttributeError, ValueError) as exc:
        raise ValueError(f"{path.name} is not an ActorCritic state_dict") from exc

    # critic input = obs_dim * n_agents + n_agents
    if critic_in % (obs_dim + 1) != 0:
        raise ValueError(f"{path.name}: critic width {critic_in} doesn't fit obs_dim {obs_dim}")
    num_agents = critic_in // (obs_dim + 1)

    net = ActorCritic(int(obs_dim), action_dim, num_agents, hidden_dim=int(hidden_dim))
    net.load_state_dict(state)
    net.eval()
    return LoadedPolicy(
        net=net,
        source=path,
        iteration=iteration,
        obs_dim=int(obs_dim),
        action_dim=action_dim,
        trained_agents=num_agents,
    )


def check_policy_fits(policy: LoadedPolicy, obs_dim: int, action_dim: int) -> None:
    """Raises a clear error if `policy` can't act in an environment with these widths."""
    if policy.obs_dim != obs_dim or policy.action_dim != action_dim:
        raise ValueError(
            f"Policy {policy.source.name} expects observation width {policy.obs_dim} and "
            f"{policy.action_dim} actions, but this environment has {obs_dim} and "
            f"{action_dim} (max_lanes / max_green_phases differ from the training setup)."
        )
