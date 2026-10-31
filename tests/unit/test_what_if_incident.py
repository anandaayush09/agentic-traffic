"""Validates the incident-aware what-if: `divert_arterial` / `meter_upstream` are judged by
simulating the incident on the real Manhattan network with and without the response, and
anything that can't be defensibly simulated stays honestly `checked=False`.
"""

from __future__ import annotations

import pytest

from atlas.cognitive.playbooks import PlaybookProposal
from atlas.cognitive.what_if import run_what_if
from atlas.common.config import project_root

NETWORK_DIR = project_root() / "data" / "networks" / "manhattan_midtown"
NET_FILE = NETWORK_DIR / "net.net.xml"
ROUTE_FILE = NETWORK_DIR / "demand" / "pm_peak.rou.xml"
VTYPES_FILE = NETWORK_DIR / "vtypes.add.xml"

pytestmark = pytest.mark.skipif(
    not (NET_FILE.is_file() and ROUTE_FILE.is_file()),
    reason="manhattan_midtown network not built",
)

JUNCTION = "42430342"


def _incident(incident_class: str = "accident") -> dict[str, object]:
    return {
        "incident_id": "whatif-1",
        "class": incident_class,
        "status": "detected",
        "confidence": 0.9,
        "location": {"link_ids": [JUNCTION]},
    }


def _what_if(proposal: PlaybookProposal, incident: dict[str, object] | None):  # type: ignore[no-untyped-def]
    return run_what_if(
        proposal, NET_FILE, ROUTE_FILE, additional_files=[VTYPES_FILE], incident=incident
    )


def _divert() -> PlaybookProposal:
    return PlaybookProposal(
        playbook_name="divert_arterial",
        params={"from_edge": JUNCTION, "to_edge": JUNCTION, "duration_s": 600.0},
        rationale="test",
    )


class TestIncidentResponseWhatIf:
    def test_divert_after_an_accident_is_really_simulated_with_numbers(self) -> None:
        result = _what_if(_divert(), _incident("accident"))

        assert result.checked is True
        assert result.metric == "mean_halted_vehicles"
        assert result.baseline_mean_pressure is not None
        assert result.proposed_mean_pressure is not None
        assert result.improved == (result.proposed_mean_pressure < result.baseline_mean_pressure)

    def test_result_is_deterministic_for_a_fixed_seed(self) -> None:
        first = _what_if(_divert(), _incident("accident"))
        second = _what_if(_divert(), _incident("accident"))

        assert first == second

    def test_meter_upstream_is_simulated_too(self) -> None:
        proposal = PlaybookProposal(
            playbook_name="meter_upstream",
            params={"corridor_id": JUNCTION, "rate": 0.7},
            rationale="test",
        )

        result = _what_if(proposal, _incident("construction"))

        assert result.checked is True
        assert result.metric == "mean_halted_vehicles"

    def test_without_the_incident_there_is_nothing_to_simulate(self) -> None:
        result = _what_if(_divert(), None)

        assert result.checked is False
        assert result.improved is None

    def test_an_incident_with_no_road_effect_is_unverifiable(self) -> None:
        result = _what_if(_divert(), _incident("sensor_fault"))

        assert result.checked is False

    def test_a_location_outside_the_network_is_unverifiable(self) -> None:
        incident = _incident("accident")
        incident["location"] = {"link_ids": ["nowhere"]}
        proposal = PlaybookProposal(
            playbook_name="divert_arterial",
            params={"from_edge": "nowhere", "to_edge": "nowhere", "duration_s": 60.0},
            rationale="test",
        )

        assert _what_if(proposal, incident).checked is False

    @pytest.mark.parametrize(
        ("name", "params"),
        [
            ("weather_capacity_derate", {"area_id": JUNCTION, "factor": 0.6}),
            ("bus_bridge", {"route_id": "r1"}),
        ],
    )
    def test_playbooks_without_a_defensible_simulation_stay_unchecked(
        self, name: str, params: dict[str, object]
    ) -> None:
        proposal = PlaybookProposal(playbook_name=name, params=params, rationale="test")

        assert _what_if(proposal, _incident("weather")).checked is False
