"""Validates PlaybookRegistry: every default playbook is retrievable by name, and
params validation genuinely enforces each playbook's own typed schema -- a playbook
cannot silently accept another playbook's parameter shape.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from atlas.cognitive.playbooks import (
    DEFAULT_PLAYBOOKS,
    DivertArterialParams,
    PlaybookProposal,
    PlaybookRegistry,
)


class TestPlaybookRegistry:
    def test_default_registry_has_between_5_and_8_playbooks(self) -> None:
        """The design's own stated range (docs/design/04_agent_architecture.md
        section 4.4: "5-8 playbooks")."""
        registry = PlaybookRegistry()
        assert 5 <= len(registry.names()) <= 8

    def test_get_returns_the_named_spec(self) -> None:
        registry = PlaybookRegistry()
        spec = registry.get("divert_arterial")
        assert spec.params_model is DivertArterialParams

    def test_get_unknown_playbook_raises(self) -> None:
        registry = PlaybookRegistry()
        with pytest.raises(KeyError, match="No playbook named"):
            registry.get("not_a_real_playbook")

    def test_every_default_playbook_is_registered_by_its_own_name(self) -> None:
        registry = PlaybookRegistry()
        for spec in DEFAULT_PLAYBOOKS:
            assert registry.get(spec.name) is spec


class TestValidateProposal:
    def test_valid_params_pass(self) -> None:
        registry = PlaybookRegistry()
        proposal = PlaybookProposal(
            playbook_name="divert_arterial",
            params={"from_edge": "A", "to_edge": "B", "duration_s": 300.0},
            rationale="test",
        )

        validated = registry.validate_proposal(proposal)

        assert isinstance(validated, DivertArterialParams)
        assert validated.duration_s == 300.0

    def test_missing_required_param_raises(self) -> None:
        registry = PlaybookRegistry()
        proposal = PlaybookProposal(
            playbook_name="divert_arterial",
            params={"from_edge": "A"},  # missing to_edge, duration_s
            rationale="test",
        )

        with pytest.raises(ValidationError):
            registry.validate_proposal(proposal)

    def test_a_playbooks_params_cannot_pass_as_a_different_playbooks_params(self) -> None:
        """The actual "bounds the blast radius" guarantee: divert_arterial's shape
        must not accidentally satisfy meter_upstream's schema or vice versa."""
        registry = PlaybookRegistry()
        # meter_upstream needs corridor_id + rate -- divert_arterial's params don't have
        # either.
        proposal = PlaybookProposal(
            playbook_name="meter_upstream",
            params={"from_edge": "A", "to_edge": "B", "duration_s": 300.0},
            rationale="test",
        )

        with pytest.raises(ValidationError):
            registry.validate_proposal(proposal)

    def test_out_of_range_param_is_rejected(self) -> None:
        registry = PlaybookRegistry()
        proposal = PlaybookProposal(
            playbook_name="meter_upstream",
            params={"corridor_id": "C1", "rate": 1.5},  # rate must be <= 1
            rationale="test",
        )

        with pytest.raises(ValidationError):
            registry.validate_proposal(proposal)

    def test_unknown_playbook_name_raises_key_error_not_validation_error(self) -> None:
        registry = PlaybookRegistry()
        proposal = PlaybookProposal(playbook_name="nonexistent", params={}, rationale="test")

        with pytest.raises(KeyError):
            registry.validate_proposal(proposal)
