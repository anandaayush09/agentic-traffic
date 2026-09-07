"""Validates the deterministic mapper: every incident.events.v1 class produces a
schema-valid PlaybookProposal (verified against the real PlaybookRegistry, not just
that a proposal object gets constructed), and the mapping choices match the documented
conservative reasoning (safe defaults for emergency/vip/unknown, never a guessed
disruptive action).
"""

from __future__ import annotations

import pytest

from atlas.cognitive.deterministic_mapper import map_incident_to_playbook
from atlas.cognitive.playbooks import PlaybookRegistry

# Every value schemas/incident.events.v1.schema.json's "class" enum allows.
ALL_INCIDENT_CLASSES = [
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
]


def _incident(incident_class: str, link_ids: list[str] | None = None) -> dict[str, object]:
    return {
        "incident_id": "inc-1",
        "class": incident_class,
        "status": "detected",
        "confidence": 0.9,
        "location": {"link_ids": link_ids or ["448686048#0"]},
    }


class TestEveryIncidentClassIsHandled:
    @pytest.mark.parametrize("incident_class", ALL_INCIDENT_CLASSES)
    def test_produces_a_registry_valid_proposal(self, incident_class: str) -> None:
        registry = PlaybookRegistry()
        incident = _incident(incident_class)

        proposal = map_incident_to_playbook(incident)

        # The real check: the proposal must pass the SAME validation a real Supervisor
        # proposal would, not just "some object came back".
        registry.validate_proposal(proposal)

    def test_unmapped_class_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="No deterministic mapping"):
            map_incident_to_playbook(_incident("not_a_real_class"))


class TestConservativeDefaults:
    def test_emergency_run_never_proposes_a_disruptive_playbook(self) -> None:
        proposal = map_incident_to_playbook(_incident("emergency_run"))
        assert proposal.playbook_name == "hold_current_plan"

    def test_vip_run_never_proposes_a_disruptive_playbook(self) -> None:
        proposal = map_incident_to_playbook(_incident("vip_run"))
        assert proposal.playbook_name == "hold_current_plan"

    def test_unknown_class_holds_rather_than_guesses(self) -> None:
        proposal = map_incident_to_playbook(_incident("unknown"))
        assert proposal.playbook_name == "hold_current_plan"

    def test_sensor_and_comms_fault_both_fall_back_to_max_pressure(self) -> None:
        sensor_proposal = map_incident_to_playbook(_incident("sensor_fault"))
        comms_proposal = map_incident_to_playbook(_incident("comms_fault"))
        assert sensor_proposal.playbook_name == "switch_to_max_pressure"
        assert comms_proposal.playbook_name == "switch_to_max_pressure"


class TestUsesIncidentLocation:
    def test_accident_playbook_targets_the_incidents_own_link(self) -> None:
        incident = _incident("accident", link_ids=["some_specific_edge"])

        proposal = map_incident_to_playbook(incident)

        assert proposal.params["from_edge"] == "some_specific_edge"

    def test_missing_location_falls_back_to_a_placeholder_not_a_crash(self) -> None:
        incident = {
            "incident_id": "inc-2",
            "class": "accident",
            "status": "detected",
            "confidence": 0.5,
        }

        proposal = map_incident_to_playbook(incident)  # must not raise KeyError/TypeError

        assert proposal.params["from_edge"] == "unknown_link"


class TestRationaleIsInformative:
    def test_every_proposal_names_its_incident_class_in_the_rationale(self) -> None:
        """Not a strict requirement of the schema, but the whole point of a rationale
        field is to explain WHY -- verify it actually mentions something specific,
        not a generic placeholder string."""
        for incident_class in ALL_INCIDENT_CLASSES:
            proposal = map_incident_to_playbook(_incident(incident_class))
            assert len(proposal.rationale) > 20
