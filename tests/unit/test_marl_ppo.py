"""Validates the PPO update: clipping behavior against hand-computed values (verified
before implementing, see ppo.py's module docstring), plus loss shapes and an end-to-end
sanity check on synthetic batches -- per the Week 5 roadmap's "PPO update unit tests
(loss shapes, clipping behavior on synthetic batches)".
"""

from __future__ import annotations

import pytest
import torch

from atlas.marl.buffer import RolloutBuffer
from atlas.marl.network import ActorCritic
from atlas.marl.ppo import PPOConfig, PPOTrainer, clipped_policy_loss, clipped_value_loss

OBS_DIM = 5
ACTION_DIM = 3
NUM_AGENTS = 2


class TestClippedPolicyLossAsymmetry:
    """Hand-computed with clip_coef=0.2 (see ppo.py's module docstring):
    ratio=2.0, advantage=+1.0 -> surr1=2.0, surr2=1.2, min=1.2, loss=-1.2 (clipped: less
    reward than the unclipped -2.0 would give).
    ratio=2.0, advantage=-1.0 -> surr1=-2.0, surr2=-1.2, min=-2.0, loss=2.0 (NOT clipped:
    identical to the unclipped penalty).
    """

    def test_positive_advantage_large_ratio_is_clipped(self) -> None:
        ratio = torch.tensor([2.0])
        advantages = torch.tensor([1.0])

        loss = clipped_policy_loss(ratio, advantages, clip_coef=0.2)

        assert loss.item() == pytest.approx(-1.2)

    def test_negative_advantage_large_ratio_is_not_clipped(self) -> None:
        ratio = torch.tensor([2.0])
        advantages = torch.tensor([-1.0])

        loss = clipped_policy_loss(ratio, advantages, clip_coef=0.2)

        assert loss.item() == pytest.approx(2.0)

    def test_ratio_within_clip_range_is_unaffected_by_clipping(self) -> None:
        # ratio=1.1 is inside [0.8, 1.2] -- surr1 == surr2, clipping has no effect.
        ratio = torch.tensor([1.1])
        advantages = torch.tensor([1.0])

        loss = clipped_policy_loss(ratio, advantages, clip_coef=0.2)

        assert loss.item() == pytest.approx(-1.1)


class TestClippedValueLoss:
    def test_matches_hand_computed_when_clipping_binds(self) -> None:
        # new_value moves 0.5 from old_value=1.0 -> 1.5, but clip_coef=0.2 caps the move
        # at 1.2. returns=2.0: unclipped err^2=(1.5-2.0)^2=0.25, clipped err^2=(1.2-2.0)^2
        # =0.64 -> loss = 0.5 * max(0.25, 0.64) = 0.32.
        new_values = torch.tensor([1.5])
        old_values = torch.tensor([1.0])
        returns = torch.tensor([2.0])

        loss = clipped_value_loss(new_values, old_values, returns, clip_coef=0.2)

        assert loss.item() == pytest.approx(0.32)

    def test_matches_unclipped_mse_when_within_clip_range(self) -> None:
        # new_value moves only 0.1 from old_value -- within clip_coef=0.2, so clipping
        # has no effect: loss = 0.5 * (new_value - returns)^2.
        new_values = torch.tensor([1.1])
        old_values = torch.tensor([1.0])
        returns = torch.tensor([2.0])

        loss = clipped_value_loss(new_values, old_values, returns, clip_coef=0.2)

        assert loss.item() == pytest.approx(0.5 * (1.1 - 2.0) ** 2)


def _make_full_buffer(num_steps: int = 4) -> RolloutBuffer:
    buf = RolloutBuffer(
        num_steps=num_steps, num_agents=NUM_AGENTS, obs_dim=OBS_DIM, action_dim=ACTION_DIM
    )
    for _ in range(num_steps):
        buf.insert(
            obs=torch.randn(NUM_AGENTS, OBS_DIM),
            joint_obs=torch.randn(NUM_AGENTS * OBS_DIM),
            actions=torch.randint(0, ACTION_DIM, (NUM_AGENTS,)),
            log_probs=torch.randn(NUM_AGENTS) * 0.1,
            rewards=torch.randn(NUM_AGENTS),
            values=torch.randn(NUM_AGENTS),
            done=False,
        )
    buf.compute_gae(last_values=torch.randn(NUM_AGENTS), last_done=False)
    return buf


class TestPPOTrainerUpdate:
    def test_raises_if_buffer_not_full(self) -> None:
        net = ActorCritic(OBS_DIM, ACTION_DIM, NUM_AGENTS)
        trainer = PPOTrainer(net)
        empty_buffer = RolloutBuffer(
            num_steps=4, num_agents=NUM_AGENTS, obs_dim=OBS_DIM, action_dim=ACTION_DIM
        )

        with pytest.raises(RuntimeError, match="must be full"):
            trainer.update(empty_buffer)

    def test_returns_finite_stats_with_expected_fields(self) -> None:
        net = ActorCritic(OBS_DIM, ACTION_DIM, NUM_AGENTS)
        trainer = PPOTrainer(net, PPOConfig(num_epochs=2, minibatch_size=4))
        buffer = _make_full_buffer()

        stats = trainer.update(buffer)

        for value in (
            stats.policy_loss,
            stats.value_loss,
            stats.entropy,
            stats.clip_fraction,
            stats.approx_kl,
        ):
            assert isinstance(value, float)
            assert value == value  # not NaN
            assert abs(value) != float("inf")

    def test_update_actually_changes_network_parameters(self) -> None:
        """The most basic possible sanity check: training does something, not nothing."""
        net = ActorCritic(OBS_DIM, ACTION_DIM, NUM_AGENTS)
        params_before = [p.clone() for p in net.parameters()]
        trainer = PPOTrainer(net, PPOConfig(num_epochs=2, minibatch_size=4))
        buffer = _make_full_buffer()

        trainer.update(buffer)

        params_after = list(net.parameters())
        assert any(
            not torch.equal(before, after)
            for before, after in zip(params_before, params_after, strict=True)
        )

    def test_respects_action_mask_during_update(self) -> None:
        """A junction with only 1 legal action out of the padded 3 (Week 4's padding
        scenario) must not crash the update -- exercises the masked path end to end,
        not just the network in isolation (test_marl_network.py already covers that)."""
        from atlas.marl.network import build_action_mask

        net = ActorCritic(OBS_DIM, ACTION_DIM, NUM_AGENTS)
        trainer = PPOTrainer(net, PPOConfig(num_epochs=1, minibatch_size=8))
        buffer = RolloutBuffer(
            num_steps=4, num_agents=NUM_AGENTS, obs_dim=OBS_DIM, action_dim=ACTION_DIM
        )
        mask = torch.stack(
            [build_action_mask(1, ACTION_DIM), build_action_mask(ACTION_DIM, ACTION_DIM)]
        )
        for _ in range(4):
            buffer.insert(
                obs=torch.randn(NUM_AGENTS, OBS_DIM),
                joint_obs=torch.randn(NUM_AGENTS * OBS_DIM),
                actions=torch.tensor([0, 1]),  # agent 0 must pick its one legal action
                log_probs=torch.randn(NUM_AGENTS) * 0.1,
                rewards=torch.randn(NUM_AGENTS),
                values=torch.randn(NUM_AGENTS),
                done=False,
                action_masks=mask,
            )
        buffer.compute_gae(last_values=torch.randn(NUM_AGENTS), last_done=False)

        stats = trainer.update(buffer)  # must not raise / produce NaN

        assert stats.policy_loss == stats.policy_loss  # not NaN
