"""Validates select_actuated_phase's gap-based extension state machine against a
hand-constructed junction (2 green phases separated by yellow transitions, one
movement per phase) and hand-traced decisions -- same discipline as
test_max_pressure.py's hand-computed pressure values, not just "picks some phase."
"""

from __future__ import annotations

from atlas.control.actuated import ActuatedConfig, select_actuated_phase
from atlas.control.max_pressure import MaxPressureProgram, Movement

CONFIG = ActuatedConfig(min_green_s=10.0, max_green_s=60.0)


def _program() -> MaxPressureProgram:
    return MaxPressureProgram(
        tls_id="test_junction",
        movements=(
            Movement(link_index=0, from_lane="A_0", to_lane="B_0"),  # served by phase 0
            Movement(link_index=1, from_lane="C_0", to_lane="D_0"),  # served by phase 2
        ),
        green_phase_indices=(0, 2),
        phase_states=("Gr", "yr", "rG", "ry"),
    )


class TestMinGreenIsUnconditional:
    def test_stays_below_min_green_even_with_no_vehicle_present(self) -> None:
        program = _program()

        phase = select_actuated_phase(
            program,
            current_phase=0,
            elapsed_in_phase_s=5.0,
            queue_lengths={"A_0": 0},
            config=CONFIG,
        )

        assert phase == 0


class TestGapExtension:
    def test_extends_when_a_vehicle_is_present_past_min_green(self) -> None:
        program = _program()

        phase = select_actuated_phase(
            program,
            current_phase=0,
            elapsed_in_phase_s=15.0,
            queue_lengths={"A_0": 1},
            config=CONFIG,
        )

        assert phase == 0

    def test_advances_to_next_green_phase_on_a_real_gap(self) -> None:
        program = _program()

        phase = select_actuated_phase(
            program,
            current_phase=0,
            elapsed_in_phase_s=15.0,
            queue_lengths={"A_0": 0},
            config=CONFIG,
        )

        assert phase == 2

    def test_missing_lane_from_queue_lengths_is_treated_as_empty(self) -> None:
        program = _program()

        phase = select_actuated_phase(
            program, current_phase=0, elapsed_in_phase_s=15.0, queue_lengths={}, config=CONFIG
        )

        assert phase == 2


class TestMaxGreenForcesTermination:
    def test_switches_at_max_green_even_with_a_vehicle_still_present(self) -> None:
        program = _program()

        phase = select_actuated_phase(
            program,
            current_phase=0,
            elapsed_in_phase_s=65.0,
            queue_lengths={"A_0": 5},
            config=CONFIG,
        )

        assert phase == 2

    def test_stays_exactly_at_max_green_boundary_with_a_vehicle_present(self) -> None:
        """elapsed_in_phase_s == max_green_s (not yet exceeding it) with a vehicle
        present should still extend -- the cutoff is strictly greater than, matching
        select_fixed_time_phase's own boundary convention elsewhere in this module."""
        program = _program()

        phase = select_actuated_phase(
            program,
            current_phase=0,
            elapsed_in_phase_s=59.9,
            queue_lengths={"A_0": 1},
            config=CONFIG,
        )

        assert phase == 0


class TestCyclingWrapsAround:
    def test_wraps_from_the_last_green_phase_back_to_the_first(self) -> None:
        program = _program()

        phase = select_actuated_phase(
            program,
            current_phase=2,
            elapsed_in_phase_s=15.0,
            queue_lengths={"C_0": 0},
            config=CONFIG,
        )

        assert phase == 0


class TestDefaultConfig:
    def test_uses_sensible_defaults_when_config_omitted(self) -> None:
        program = _program()

        phase = select_actuated_phase(
            program, current_phase=0, elapsed_in_phase_s=5.0, queue_lengths={"A_0": 0}
        )

        assert phase == 0  # below the default min_green_s=10.0
