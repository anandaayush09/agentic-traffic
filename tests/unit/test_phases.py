from __future__ import annotations

import pytest

from atlas.control.phases import green_phase_indices, is_yellow_phase, yellow_phase_after


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


class TestYellowPhaseAfter:
    # Copied verbatim from junction B1 in the sandbox network.
    STATES = ["GGggrrrrGGggrrrr", "yyyyrrrryyyyrrrr", "rrrrGGggrrrrGGgg", "rrrryyyyrrrryyyy"]

    def test_finds_yellow_after_first_green(self) -> None:
        assert yellow_phase_after(self.STATES, 0) == 1

    def test_finds_yellow_after_second_green(self) -> None:
        assert yellow_phase_after(self.STATES, 2) == 3

    def test_wraps_around_cyclically(self) -> None:
        states = ["yyyy", "GGrr"]  # green is the LAST phase; its yellow wraps to index 0
        assert yellow_phase_after(states, 1) == 0

    def test_raises_if_next_phase_is_not_yellow(self) -> None:
        states = ["GGrr", "rrGG"]  # no yellow phases at all

        with pytest.raises(ValueError, match="is not"):
            yellow_phase_after(states, 0)
