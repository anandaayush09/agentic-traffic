"""Validates the scripted incident suite's scoring logic against hand-constructed
cases (grounding must actually catch a fabricated location claim, not just pass
everything), then runs the real 10-incident suite end to end through the real
deterministic mapper and real playbook registry -- no mocks -- confirming the roadmap's
own targets (playbook accuracy, grounding rate) hold for the terminal fallback path.
"""

from __future__ import annotations

from atlas.cognitive.incident_suite import (
    SCRIPTED_INCIDENTS,
    ScriptedIncident,
    is_grounded,
    run_suite,
    score_incident,
)
from atlas.cognitive.playbooks import PlaybookProposal, PlaybookRegistry
from atlas.cognitive.provider_chain import ProviderChain

_EDGE = "448686048#0"
_FABRICATED_EDGE = "999999999#0"


def _incident_with_location(*link_ids: str) -> ScriptedIncident:
    return ScriptedIncident(
        incident_id="test-incident",
        incident={
            "msg_id": "00000000-0000-0000-0000-0000000000ff",
            "ts": "2026-01-01T00:00:00Z",
            "schema_version": "1",
            "incident_id": "test-incident",
            "class": "accident",
            "status": "detected",
            "location": {"link_ids": list(link_ids)},
            "confidence": 0.9,
        },
        expected_playbook="divert_arterial",
    )


class TestIsGrounded:
    def test_proposal_citing_the_incidents_own_edge_is_grounded(self) -> None:
        scripted = _incident_with_location(_EDGE)
        proposal = PlaybookProposal(
            playbook_name="divert_arterial",
            params={"from_edge": _EDGE, "to_edge": _EDGE, "duration_s": 600.0},
            rationale="test",
        )

        assert is_grounded(scripted, proposal) is True

    def test_proposal_citing_a_fabricated_edge_is_not_grounded(self) -> None:
        """The core thing this metric must catch: a claim about a location the
        incident never reported -- a hallucinated alternate route, concretely."""
        scripted = _incident_with_location(_EDGE)
        proposal = PlaybookProposal(
            playbook_name="divert_arterial",
            params={"from_edge": _EDGE, "to_edge": _FABRICATED_EDGE, "duration_s": 600.0},
            rationale="test",
        )

        assert is_grounded(scripted, proposal) is False

    def test_hold_current_plan_reason_text_is_not_treated_as_a_location_claim(self) -> None:
        scripted = _incident_with_location(_EDGE)
        proposal = PlaybookProposal(
            playbook_name="hold_current_plan",
            params={"reason": "no safe action without more evidence"},
            rationale="test",
        )

        assert is_grounded(scripted, proposal) is True

    def test_no_location_claim_is_trivially_grounded_even_with_no_incident_location(
        self,
    ) -> None:
        scripted = ScriptedIncident(
            incident_id="test-unknown",
            incident={
                "msg_id": "00000000-0000-0000-0000-0000000000fe",
                "ts": "2026-01-01T00:00:00Z",
                "schema_version": "1",
                "incident_id": "test-unknown",
                "class": "unknown",
                "status": "detected",
                "confidence": 0.3,
            },
            expected_playbook="hold_current_plan",
        )
        proposal = PlaybookProposal(
            playbook_name="hold_current_plan", params={"reason": "unclear"}, rationale="test"
        )

        assert is_grounded(scripted, proposal) is True


class TestScoreIncident:
    def test_correct_grounded_valid_proposal_scores_all_true(self) -> None:
        scripted = _incident_with_location(_EDGE)
        proposal = PlaybookProposal(
            playbook_name="divert_arterial",
            params={"from_edge": _EDGE, "to_edge": _EDGE, "duration_s": 600.0},
            rationale="test",
        )
        registry = PlaybookRegistry()

        score = score_incident(scripted, proposal, "deterministic", registry)

        assert score.playbook_correct is True
        assert score.grounded is True
        assert score.params_valid is True

    def test_wrong_playbook_scores_playbook_correct_false(self) -> None:
        scripted = _incident_with_location(_EDGE)
        proposal = PlaybookProposal(
            playbook_name="hold_current_plan", params={"reason": "wrong choice"}, rationale="test"
        )
        registry = PlaybookRegistry()

        score = score_incident(scripted, proposal, "deterministic", registry)

        assert score.playbook_correct is False

    def test_params_that_fail_the_playbooks_own_model_score_params_valid_false(self) -> None:
        scripted = _incident_with_location(_EDGE)
        proposal = PlaybookProposal(
            playbook_name="divert_arterial",
            params={"from_edge": _EDGE},  # missing required to_edge/duration_s
            rationale="test",
        )
        registry = PlaybookRegistry()

        score = score_incident(scripted, proposal, "deterministic", registry)

        assert score.params_valid is False


class TestRunSuiteEndToEnd:
    def test_ten_incidents_covering_every_incident_class(self) -> None:
        classes = {incident.incident["class"] for incident in SCRIPTED_INCIDENTS}

        assert len(SCRIPTED_INCIDENTS) == 10
        assert classes == {
            "accident",
            "closure",
            "construction",
            "weather",
            "event_surge",
            "sensor_fault",
            "comms_fault",
            "emergency_run",
            "vip_run",
            "unknown",
        }

    def test_real_deterministic_fallback_meets_the_roadmap_targets(self) -> None:
        """No LLM providers are configured (no API keys in this environment, same
        stance as provider_chain.py) -- every incident here runs through the REAL
        deterministic mapper and REAL playbook registry validation, no mocks. This
        validates the terminal fallback itself matches what a human operator would
        pick for each class-level default (docs/design/01_problem_definition.md's
        target: >= 80% correct playbook selection), not an LLM's judgment -- that
        comparison needs real provider keys this environment doesn't have.
        """
        chain = ProviderChain(providers=[])
        registry = PlaybookRegistry()

        summary = run_suite(chain, registry)

        assert len(summary.scores) == 10
        assert all(score.provider_used == "deterministic" for score in summary.scores)
        assert summary.playbook_accuracy >= 0.80
        assert summary.grounding_rate == 1.0
        assert summary.param_sanity_rate == 1.0
