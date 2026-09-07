"""Validates run_what_if against the real sandbox network -- the actual simulation
comparison, not a mock, for the two playbook types this project can genuinely model.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from atlas.cognitive.playbooks import PlaybookProposal
from atlas.cognitive.what_if import run_what_if


@pytest.fixture(scope="session")
def sandbox_net_route() -> tuple[Path, Path]:
    import sys

    from atlas.common.config import project_root

    sys.path.insert(0, str(project_root() / "scripts"))
    import generate_sandbox_network

    generate_sandbox_network.generate()
    return generate_sandbox_network.NET_FILE, generate_sandbox_network.ROUTE_FILE


class TestHoldCurrentPlan:
    def test_is_always_checked_and_improved_without_touching_sumo(
        self, sandbox_net_route: tuple[Path, Path]
    ) -> None:
        net_file, route_file = sandbox_net_route
        proposal = PlaybookProposal(
            playbook_name="hold_current_plan", params={"reason": "test"}, rationale="test"
        )

        result = run_what_if(proposal, net_file, route_file)

        assert result.checked is True
        assert result.improved is True


class TestSwitchToMaxPressure:
    def test_runs_a_real_comparison_against_the_sandbox_network(
        self, sandbox_net_route: tuple[Path, Path]
    ) -> None:
        net_file, route_file = sandbox_net_route
        proposal = PlaybookProposal(
            playbook_name="switch_to_max_pressure",
            params={"junction_ids": ["B1"]},
            rationale="test",
        )

        result = run_what_if(proposal, net_file, route_file, num_decisions=15)

        assert result.checked is True
        assert result.improved is not None
        assert result.baseline_mean_pressure is not None
        assert result.proposed_mean_pressure is not None

    def test_missing_junction_ids_is_reported_as_unchecked_not_a_crash(
        self, sandbox_net_route: tuple[Path, Path]
    ) -> None:
        net_file, route_file = sandbox_net_route
        proposal = PlaybookProposal(
            playbook_name="switch_to_max_pressure", params={}, rationale="test"
        )

        result = run_what_if(proposal, net_file, route_file)

        assert result.checked is False
        assert result.improved is None


class TestUnsimulatablePlaybooks:
    @pytest.mark.parametrize(
        "playbook_name,params",
        [
            ("divert_arterial", {"from_edge": "A", "to_edge": "B", "duration_s": 300.0}),
            ("meter_upstream", {"corridor_id": "C1", "rate": 0.5}),
            ("bus_bridge", {"route_id": "R1"}),
            ("weather_capacity_derate", {"area_id": "A1", "factor": 0.5}),
        ],
    )
    def test_reports_unchecked_honestly_rather_than_fabricating_an_effect(
        self, sandbox_net_route: tuple[Path, Path], playbook_name: str, params: dict[str, object]
    ) -> None:
        net_file, route_file = sandbox_net_route
        proposal = PlaybookProposal(playbook_name=playbook_name, params=params, rationale="test")

        result = run_what_if(proposal, net_file, route_file)

        assert result.checked is False
        assert result.improved is None
