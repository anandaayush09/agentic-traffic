"""Validates RolloutBuffer against hand-computed GAE values (not just "does it run") --
per the Week 5 roadmap's testing guidance, the same standard this project has applied to
every other piece of math (max-pressure, Webster's formula, the shield).
"""

from __future__ import annotations

import pytest
import torch

from atlas.marl.buffer import RolloutBuffer

GAMMA = 0.9
GAE_LAMBDA = 0.95


class TestConstruction:
    def test_rejects_nonpositive_dims(self) -> None:
        with pytest.raises(ValueError, match="must all be >= 1"):
            RolloutBuffer(num_steps=0, num_agents=1, obs_dim=1, action_dim=1)

    def test_starts_not_full(self) -> None:
        buf = RolloutBuffer(num_steps=2, num_agents=1, obs_dim=3, action_dim=2)

        assert buf.full is False


def _insert_step(
    buf: RolloutBuffer,
    num_agents: int,
    obs_dim: int,
    action_dim: int,
    reward: list[float],
    value: list[float],
    done: bool,
) -> None:
    buf.insert(
        obs=torch.zeros(num_agents, obs_dim),
        joint_obs=torch.zeros(num_agents * obs_dim),
        actions=torch.zeros(num_agents, dtype=torch.long),
        log_probs=torch.zeros(num_agents),
        rewards=torch.tensor(reward),
        values=torch.tensor(value),
        done=done,
    )


class TestInsertAndReset:
    def test_becomes_full_after_num_steps_inserts(self) -> None:
        buf = RolloutBuffer(num_steps=2, num_agents=1, obs_dim=1, action_dim=1)
        _insert_step(buf, 1, 1, 1, [1.0], [0.5], False)
        assert buf.full is False
        _insert_step(buf, 1, 1, 1, [1.0], [0.5], False)
        assert buf.full is True

    def test_insert_past_capacity_raises(self) -> None:
        buf = RolloutBuffer(num_steps=1, num_agents=1, obs_dim=1, action_dim=1)
        _insert_step(buf, 1, 1, 1, [1.0], [0.5], False)

        with pytest.raises(RuntimeError, match="Buffer is full"):
            _insert_step(buf, 1, 1, 1, [1.0], [0.5], False)

    def test_reset_allows_new_inserts(self) -> None:
        buf = RolloutBuffer(num_steps=1, num_agents=1, obs_dim=1, action_dim=1)
        _insert_step(buf, 1, 1, 1, [1.0], [0.5], False)
        buf.reset()

        assert buf.full is False
        _insert_step(buf, 1, 1, 1, [1.0], [0.5], False)  # must not raise
        assert buf.full is True


class TestComputeGaeRequiresFullBuffer:
    def test_raises_if_not_full(self) -> None:
        buf = RolloutBuffer(num_steps=2, num_agents=1, obs_dim=1, action_dim=1)
        _insert_step(buf, 1, 1, 1, [1.0], [0.5], False)

        with pytest.raises(RuntimeError, match="not full"):
            buf.compute_gae(last_values=torch.tensor([0.0]), last_done=False)


class TestGaeHandComputed:
    """T=2, N=1: rewards=[1.0, 2.0], values=[0.5, 0.6], dones=[0,0], last_values=[0.7].

    t=1: next_non_terminal=1, next_values=0.7
         delta[1] = 2.0 + 0.9*0.7 - 0.6 = 2.03
         advantages[1] = 2.03
    t=0: next_non_terminal=1, next_values=values[1]=0.6
         delta[0] = 1.0 + 0.9*0.6 - 0.5 = 1.04
         advantages[0] = 1.04 + 0.9*0.95*2.03 = 1.04 + 1.73565 = 2.77565

    returns = advantages + values -> [3.27565, 2.63]
    """

    def test_single_agent_advantages_and_returns(self) -> None:
        buf = RolloutBuffer(num_steps=2, num_agents=1, obs_dim=1, action_dim=1)
        _insert_step(buf, 1, 1, 1, [1.0], [0.5], False)
        _insert_step(buf, 1, 1, 1, [2.0], [0.6], False)

        buf.compute_gae(
            last_values=torch.tensor([0.7]), last_done=False, gamma=GAMMA, gae_lambda=GAE_LAMBDA
        )

        assert buf.advantages[:, 0].tolist() == pytest.approx([2.77565, 2.03], abs=1e-5)
        assert buf.returns[:, 0].tolist() == pytest.approx([3.27565, 2.63], abs=1e-5)

    def test_two_agents_computed_independently(self) -> None:
        """Agent 1's own hand-computed sequence (rewards=[0.5,1.5], values=[0.2,0.3],
        last_value=0.4):
        t=1: delta=1.5+0.9*0.4-0.3=1.56; advantages[1]=1.56
        t=0: delta=0.5+0.9*0.3-0.2=0.57; advantages[0]=0.57+0.9*0.95*1.56=1.9038
        returns -> [2.1038, 1.86]
        Must match agent 0's independently-computed values from the single-agent test,
        proving no cross-agent leakage in the per-column recursion.
        """
        buf = RolloutBuffer(num_steps=2, num_agents=2, obs_dim=1, action_dim=1)
        _insert_step(buf, 2, 1, 1, [1.0, 0.5], [0.5, 0.2], False)
        _insert_step(buf, 2, 1, 1, [2.0, 1.5], [0.6, 0.3], False)

        buf.compute_gae(
            last_values=torch.tensor([0.7, 0.4]),
            last_done=False,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
        )

        assert buf.advantages[:, 0].tolist() == pytest.approx([2.77565, 2.03], abs=1e-5)
        assert buf.returns[:, 0].tolist() == pytest.approx([3.27565, 2.63], abs=1e-5)
        assert buf.advantages[:, 1].tolist() == pytest.approx([1.9038, 1.56], abs=1e-4)
        assert buf.returns[:, 1].tolist() == pytest.approx([2.1038, 1.86], abs=1e-4)


class TestGaeWithDoneFlag:
    def test_terminal_step_does_not_crash_and_stays_finite(self) -> None:
        """This project's actual environment (MultiAgentSignalEnv) never sets done=True
        (Week 4's documented design -- episode length is the training script's concern).
        This is a basic robustness check for the path, not a hand-verified boundary
        case: done handling here follows the standard CleanRL/SB3 convention faithfully,
        but isn't exercised by this project's real usage.
        """
        buf = RolloutBuffer(num_steps=3, num_agents=1, obs_dim=1, action_dim=1)
        _insert_step(buf, 1, 1, 1, [1.0], [0.5], False)
        _insert_step(buf, 1, 1, 1, [1.0], [0.5], True)
        _insert_step(buf, 1, 1, 1, [1.0], [0.5], False)

        buf.compute_gae(last_values=torch.tensor([0.5]), last_done=False)

        assert torch.all(torch.isfinite(buf.advantages))
        assert torch.all(torch.isfinite(buf.returns))
