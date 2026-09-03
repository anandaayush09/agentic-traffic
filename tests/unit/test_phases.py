from __future__ import annotations

from atlas.control.phases import green_phase_indices, is_yellow_phase


class TestIsYellowPhase:
    def test_detects_lowercase_y(self) -> None:
        assert is_yellow_phase("rryy") is True

    def test_detects_uppercase_y(self) -> None:
        assert is_yellow_phase("rrYY") is True

    def test_pure_green_red_is_not_yellow(self) -> None:
        assert is_yellow_phase("GGrr") is False
        assert is_yellow_phase("ggrr") is False


class TestGreenPhaseIndices:
    def test_identifies_green_phases_from_real_states(self) -> None:
        # Copied verbatim from a real junction (42428283) in the built network.
        states = ["GGrr", "yyrr", "rrGG", "rryy"]

        assert green_phase_indices(states) == (0, 2)

    def test_empty_when_all_yellow(self) -> None:
        assert green_phase_indices(["yyyy", "YYYY"]) == ()

    def test_empty_input(self) -> None:
        assert green_phase_indices([]) == ()
