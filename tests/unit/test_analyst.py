"""Validates the Analyst/XAI agent v1: explanations genuinely quote logged values (not
templated placeholders), refusal is a real distinct exception for an unlogged decision,
and the counterfactual probe re-runs a real ActorCritic policy rather than guessing.
"""

from __future__ import annotations

import pytest
import torch

from atlas.cognitive.analyst import (
    AnalystAgent,
    DecisionLogEntry,
    DecisionLogStore,
    DecisionNotFoundError,
    explain_decision,
    run_counterfactual,
)
from atlas.marl.network import ActorCritic

OBS_DIM = 6
ACTION_DIM = 3
NUM_AGENTS = 1


def _shield_modified_entry() -> DecisionLogEntry:
    return DecisionLogEntry(
        decision_id="d1",
        intersection_id="42430333",
        actor="rl_agent",
        phase_id=0,
        shield_modified=True,
        shield_reason="min_green",
        pressure=12.5,
    )


def _shield_unmodified_entry() -> DecisionLogEntry:
    return DecisionLogEntry(
        decision_id="d2",
        intersection_id="42430333",
        actor="max_pressure",
        phase_id=1,
        shield_modified=False,
        shield_reason=None,
        pressure=-3.0,
    )


class TestExplainDecision:
    def test_shield_modified_explanation_quotes_the_real_reason(self) -> None:
        explanation = explain_decision(_shield_modified_entry())

        assert "42430333" in explanation
        assert "min_green" in explanation
        assert "OVERRIDDEN" in explanation
        assert "12.50" in explanation

    def test_shield_unmodified_explanation_says_shield_did_not_intervene(self) -> None:
        explanation = explain_decision(_shield_unmodified_entry())

        assert "did not need to intervene" in explanation
        assert "max_pressure" in explanation
        assert "-3.00" in explanation


class TestDecisionLogStore:
    def test_records_and_retrieves_by_id(self) -> None:
        store = DecisionLogStore()
        entry = _shield_modified_entry()
        store.record(entry)

        assert store.get("d1") is entry

    def test_unlogged_decision_raises_decision_not_found(self) -> None:
        store = DecisionLogStore()
        with pytest.raises(DecisionNotFoundError, match="never recorded"):
            store.get("does-not-exist")


class TestAnalystAgentExplain:
    def test_explain_delegates_to_the_log_store(self) -> None:
        store = DecisionLogStore()
        store.record(_shield_modified_entry())
        agent = AnalystAgent(store)

        explanation = agent.explain("d1")

        assert "min_green" in explanation

    def test_explain_refuses_for_an_unlogged_decision_rather_than_guessing(self) -> None:
        agent = AnalystAgent(DecisionLogStore())

        with pytest.raises(DecisionNotFoundError):
            agent.explain("unknown-id")


class TestCounterfactualProbe:
    def test_real_perturbation_changes_the_chosen_action_after_training(self) -> None:
        """Trains the actor on a tiny supervised task where feature 0's sign alone
        determines the correct action (negative -> action 2, positive -> action 0),
        then verifies the counterfactual probe detects a REAL re-evaluation of the
        learned policy flipping its choice -- not a hand-guessed weight construction.
        """
        torch.manual_seed(0)
        net = ActorCritic(obs_dim=OBS_DIM, action_dim=ACTION_DIM, num_agents=NUM_AGENTS)
        optimizer = torch.optim.Adam(net.actor.parameters(), lr=1e-2)

        for _ in range(300):
            obs = torch.zeros(16, OBS_DIM)
            obs[:8, 0] = -1.0  # first half: negative feature 0 -> label action 2
            obs[8:, 0] = 1.0  # second half: positive feature 0 -> label action 0
            labels = torch.cat([torch.full((8,), 2), torch.full((8,), 0)])

            logits = net.action_logits(obs)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()

        net.eval()
        original_obs = torch.zeros(OBS_DIM)
        original_obs[0] = -1.0

        result = run_counterfactual(net, original_obs, feature_index=0, new_value=1.0)

        assert result.original_action == 2
        assert result.perturbed_action == 0
        assert result.action_changed is True
        assert "CHANGES" in result.description

    def test_rejects_out_of_range_feature_index(self) -> None:
        net = ActorCritic(obs_dim=OBS_DIM, action_dim=ACTION_DIM, num_agents=NUM_AGENTS)
        obs = torch.zeros(OBS_DIM)

        with pytest.raises(ValueError, match="out of range"):
            run_counterfactual(net, obs, feature_index=OBS_DIM, new_value=1.0)

    def test_no_change_reports_action_unchanged(self) -> None:
        net = ActorCritic(obs_dim=OBS_DIM, action_dim=ACTION_DIM, num_agents=NUM_AGENTS)
        obs = torch.zeros(OBS_DIM)

        result = run_counterfactual(net, obs, feature_index=0, new_value=0.0)  # no real change

        assert result.action_changed is False
        assert result.original_action == result.perturbed_action
