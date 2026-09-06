"""Validates the checkpoint save/load round trip and config_hash determinism -- the
Week 6 "fully resumable" requirement (docs/design/06_ai_pipeline.md section 6.1) --
without needing SUMO/libsumo, since none of this logic touches the environment.
"""

from __future__ import annotations

import torch

from atlas.marl.network import ActorCritic
from atlas.marl.ppo import PPOTrainer
from atlas.marl.training import (
    RunningRewardNormalizer,
    config_hash,
    load_checkpoint,
    save_checkpoint,
)

OBS_DIM = 4
ACTION_DIM = 2
NUM_AGENTS = 3


class TestConfigHash:
    def test_same_payload_same_hash(self) -> None:
        payload = {"tls_ids": ["a", "b"], "seed": 42}
        assert config_hash(payload) == config_hash(payload)

    def test_key_order_does_not_affect_hash(self) -> None:
        assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})

    def test_different_payload_different_hash(self) -> None:
        assert config_hash({"seed": 1}) != config_hash({"seed": 2})

    def test_hash_is_short_and_deterministic_across_calls(self) -> None:
        h = config_hash({"x": 1})
        assert len(h) == 8
        assert h == config_hash({"x": 1})


class TestRunningRewardNormalizerStateDict:
    def test_round_trips_exactly(self) -> None:
        normalizer = RunningRewardNormalizer()
        normalizer.update(torch.tensor([1.0, 2.0, 3.0]))
        normalizer.update(torch.tensor([-4.0, 5.0]))

        restored = RunningRewardNormalizer()
        restored.load_state_dict(normalizer.state_dict())

        assert restored.mean == normalizer.mean
        assert restored.var == normalizer.var
        assert restored.count == normalizer.count


class TestCheckpointRoundTrip:
    def test_save_then_load_restores_iteration_and_weights(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        net = ActorCritic(OBS_DIM, ACTION_DIM, NUM_AGENTS)
        trainer = PPOTrainer(net)
        normalizer = RunningRewardNormalizer()
        normalizer.update(torch.tensor([1.0, -2.0, 0.5]))

        path = tmp_path / "run_latest.pt"
        save_checkpoint(
            path, iteration=7, net=net, optimizer=trainer.optimizer, reward_normalizer=normalizer
        )

        restored_net = ActorCritic(OBS_DIM, ACTION_DIM, NUM_AGENTS)
        restored_trainer = PPOTrainer(restored_net)
        restored_normalizer = RunningRewardNormalizer()

        last_iteration = load_checkpoint(
            path, restored_net, restored_trainer.optimizer, restored_normalizer
        )

        assert last_iteration == 7
        assert restored_normalizer.mean == normalizer.mean
        assert restored_normalizer.var == normalizer.var
        for original, restored in zip(net.parameters(), restored_net.parameters(), strict=True):
            assert torch.equal(original, restored)

    def test_load_restores_optimizer_momentum_not_just_weights(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """A checkpoint that only restored weights would silently reset Adam's
        momentum/variance buffers on resume, changing training dynamics -- verify the
        optimizer's own state dict round-trips too, not just the network."""
        net = ActorCritic(OBS_DIM, ACTION_DIM, NUM_AGENTS)
        trainer = PPOTrainer(net)
        # Take one real optimizer step so Adam actually has momentum state to lose.
        loss = net.actor(torch.randn(4, OBS_DIM)).sum()
        loss.backward()
        trainer.optimizer.step()
        normalizer = RunningRewardNormalizer()

        path = tmp_path / "run_latest.pt"
        save_checkpoint(
            path, iteration=0, net=net, optimizer=trainer.optimizer, reward_normalizer=normalizer
        )

        restored_net = ActorCritic(OBS_DIM, ACTION_DIM, NUM_AGENTS)
        restored_trainer = PPOTrainer(restored_net)
        load_checkpoint(path, restored_net, restored_trainer.optimizer, RunningRewardNormalizer())

        original_state = trainer.optimizer.state_dict()["state"]
        restored_state = restored_trainer.optimizer.state_dict()["state"]
        assert original_state.keys() == restored_state.keys()
        assert len(original_state) > 0  # sanity: Adam actually accumulated state
