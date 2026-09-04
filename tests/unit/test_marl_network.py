"""Validates the MAPPO actor-critic network against synthetic data (no SUMO/twin
dependency -- pure PyTorch), per the Week 5 roadmap's "PPO update unit tests (loss
shapes, clipping behavior on synthetic batches)". Action masking is checked against a
real degenerate case (a single legal action out of a padded space), not just the common
case, since that's exactly where a naive -inf-masking implementation tends to produce NaN.
"""

from __future__ import annotations

import math

import pytest
import torch

from atlas.marl.network import ActorCritic, build_action_mask

OBS_DIM = 19  # matches Week 4's sandbox observation_dim (4*4 + 2 + 1)
ACTION_DIM = 3  # padded beyond the sandbox's real 2, to exercise masking
NUM_AGENTS = 4


class TestBuildActionMask:
    def test_marks_legal_slots_true_and_padding_false(self) -> None:
        mask = build_action_mask(num_legal_actions=2, max_actions=3)

        assert mask.tolist() == [True, True, False]

    def test_full_mask_when_no_padding(self) -> None:
        mask = build_action_mask(num_legal_actions=3, max_actions=3)

        assert mask.tolist() == [True, True, True]

    def test_rejects_zero_legal_actions(self) -> None:
        with pytest.raises(ValueError, match="must be >= 1"):
            build_action_mask(num_legal_actions=0, max_actions=3)

    def test_rejects_legal_actions_exceeding_max(self) -> None:
        with pytest.raises(ValueError, match="must be <="):
            build_action_mask(num_legal_actions=4, max_actions=3)


class TestActorCriticConstruction:
    def test_rejects_nonpositive_dims(self) -> None:
        with pytest.raises(ValueError, match="must all be >= 1"):
            ActorCritic(obs_dim=0, action_dim=ACTION_DIM, num_agents=NUM_AGENTS)


class TestActionLogitsMasking:
    def test_unmasked_logits_pass_through_unchanged(self) -> None:
        net = ActorCritic(OBS_DIM, ACTION_DIM, NUM_AGENTS)
        obs = torch.randn(1, OBS_DIM)

        with_none = net.action_logits(obs, action_mask=None)
        full_mask = torch.ones(1, ACTION_DIM, dtype=torch.bool)
        with_full_mask = net.action_logits(obs, action_mask=full_mask)

        assert torch.equal(with_none, with_full_mask)

    def test_masked_positions_become_negative_infinity(self) -> None:
        net = ActorCritic(OBS_DIM, ACTION_DIM, NUM_AGENTS)
        obs = torch.randn(1, OBS_DIM)
        mask = torch.tensor([[True, True, False]])

        logits = net.action_logits(obs, action_mask=mask)

        assert logits[0, 2].item() == float("-inf")
        assert math.isfinite(logits[0, 0].item())
        assert math.isfinite(logits[0, 1].item())


class TestActSampling:
    def test_sampled_actions_are_always_within_range(self) -> None:
        net = ActorCritic(OBS_DIM, ACTION_DIM, NUM_AGENTS)
        obs = torch.randn(50, OBS_DIM)

        action, log_prob, entropy = net.act(obs)

        assert action.shape == (50,)
        assert torch.all(action >= 0) and torch.all(action < ACTION_DIM)
        assert torch.all(torch.isfinite(log_prob))
        assert torch.all(torch.isfinite(entropy))

    def test_deterministic_matches_argmax(self) -> None:
        net = ActorCritic(OBS_DIM, ACTION_DIM, NUM_AGENTS)
        obs = torch.randn(10, OBS_DIM)

        action, _, _ = net.act(obs, deterministic=True)
        expected = torch.argmax(net.action_logits(obs), dim=-1)

        assert torch.equal(action, expected)

    def test_masked_action_never_samples_padding_across_many_trials(self) -> None:
        """The degenerate case: 2 legal actions out of a padded 3. Sampled 500 times --
        must NEVER pick the masked-out action, and log_prob/entropy must stay finite
        every time (a naive implementation can silently produce NaN here)."""
        net = ActorCritic(OBS_DIM, ACTION_DIM, NUM_AGENTS)
        obs = torch.randn(1, OBS_DIM)
        mask = build_action_mask(num_legal_actions=2, max_actions=ACTION_DIM).unsqueeze(0)

        for _ in range(500):
            action, log_prob, entropy = net.act(obs, action_mask=mask)
            assert action.item() in (0, 1)
            assert math.isfinite(log_prob.item())
            assert math.isfinite(entropy.item())

    def test_single_legal_action_has_zero_entropy_and_always_selects_it(self) -> None:
        """The real degenerate case referenced in network.py's docstring: exactly one
        legal action. The distribution collapses to a single outcome -- entropy must be
        exactly 0 (not NaN), and every sample must be that one action."""
        net = ActorCritic(OBS_DIM, ACTION_DIM, NUM_AGENTS)
        obs = torch.randn(1, OBS_DIM)
        mask = build_action_mask(num_legal_actions=1, max_actions=ACTION_DIM).unsqueeze(0)

        for _ in range(20):
            action, log_prob, entropy = net.act(obs, action_mask=mask)
            assert action.item() == 0
            assert math.isfinite(log_prob.item())
            assert entropy.item() == pytest.approx(0.0, abs=1e-5)


class TestEvaluateActions:
    def test_returns_correctly_shaped_batch_outputs(self) -> None:
        net = ActorCritic(OBS_DIM, ACTION_DIM, NUM_AGENTS)
        batch_size = 8
        obs = torch.randn(batch_size, OBS_DIM)
        joint_obs = torch.randn(batch_size, OBS_DIM * NUM_AGENTS)
        agent_id = torch.randint(0, NUM_AGENTS, (batch_size,))
        action = torch.randint(0, ACTION_DIM, (batch_size,))

        log_prob, entropy, value = net.evaluate_actions(obs, joint_obs, agent_id, action)

        assert log_prob.shape == (batch_size,)
        assert entropy.shape == (batch_size,)
        assert value.shape == (batch_size,)
        assert torch.all(torch.isfinite(log_prob))
        assert torch.all(torch.isfinite(entropy))
        assert torch.all(torch.isfinite(value))

    def test_value_depends_on_joint_obs_not_local_obs(self) -> None:
        """A key CTDE correctness property: two calls with the SAME local obs but
        DIFFERENT joint_obs must produce different values (the critic must actually be
        using the joint observation, not silently ignoring it)."""
        net = ActorCritic(OBS_DIM, ACTION_DIM, NUM_AGENTS)
        obs = torch.randn(1, OBS_DIM)
        joint_obs_a = torch.randn(1, OBS_DIM * NUM_AGENTS)
        joint_obs_b = torch.randn(1, OBS_DIM * NUM_AGENTS)
        agent_id = torch.zeros(1, dtype=torch.long)
        action = torch.zeros(1, dtype=torch.long)

        _, _, value_a = net.evaluate_actions(obs, joint_obs_a, agent_id, action)
        _, _, value_b = net.evaluate_actions(obs, joint_obs_b, agent_id, action)

        assert value_a.item() != pytest.approx(value_b.item())

    def test_value_depends_on_agent_id(self) -> None:
        """The whole point of the per-agent-conditioned critic: the SAME joint_obs must
        produce DIFFERENT values for different agent_id -- otherwise this would just be
        a single shared team value with extra steps."""
        net = ActorCritic(OBS_DIM, ACTION_DIM, NUM_AGENTS)
        joint_obs = torch.randn(1, OBS_DIM * NUM_AGENTS)

        value_agent_0 = net.get_value(joint_obs, torch.tensor([0]))
        value_agent_1 = net.get_value(joint_obs, torch.tensor([1]))

        assert value_agent_0.item() != pytest.approx(value_agent_1.item())

    def test_get_value_matches_evaluate_actions_value(self) -> None:
        net = ActorCritic(OBS_DIM, ACTION_DIM, NUM_AGENTS)
        joint_obs = torch.randn(1, OBS_DIM * NUM_AGENTS)
        obs = torch.randn(1, OBS_DIM)
        agent_id = torch.tensor([2])
        action = torch.zeros(1, dtype=torch.long)

        _, _, value_from_evaluate = net.evaluate_actions(obs, joint_obs, agent_id, action)
        value_from_get_value = net.get_value(joint_obs, agent_id)

        assert torch.equal(value_from_evaluate, value_from_get_value)
