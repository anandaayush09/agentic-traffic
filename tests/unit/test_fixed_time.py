from __future__ import annotations

import sys
from pathlib import Path

import pytest

from atlas.common.config import project_root
from atlas.control.fixed_time import (
    compute_webster_green_splits,
    native_green_durations_from_net_file,
    select_fixed_time_phase,
)


class TestComputeWebsterGreenSplits:
    def test_hand_computed_two_phase_example(self) -> None:
        # y = [0.3, 0.2] -> Y = 0.5, L = 4*2 = 8
        # C0 = (1.5*8 + 5) / (1 - 0.5) = 17 / 0.5 = 34.0
        # effective green = 34 - 8 = 26.0
        # green_1 = (0.3/0.5)*26 = 15.6, green_2 = (0.2/0.5)*26 = 10.4
        result = compute_webster_green_splits(
            flow_ratios=[0.3, 0.2], lost_time_per_phase_s=4.0, min_green_s=5.0
        )

        assert result.cycle_s == pytest.approx(34.0)
        assert result.green_durations_s == pytest.approx((15.6, 10.4))
        assert sum(result.green_durations_s) == pytest.approx(26.0)

    def test_min_green_floor_applies_to_low_flow_phase(self) -> None:
        # A phase with a tiny flow ratio would otherwise compute to less than the safe
        # minimum green -- the floor must override the proportional split for it alone.
        result = compute_webster_green_splits(
            flow_ratios=[0.45, 0.02], lost_time_per_phase_s=4.0, min_green_s=10.0
        )

        assert result.green_durations_s[1] == 10.0  # floored
        assert result.green_durations_s[0] > 10.0  # unaffected, still proportional

    def test_rejects_empty_flow_ratios(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            compute_webster_green_splits(flow_ratios=[])

    def test_rejects_nonpositive_flow_ratio(self) -> None:
        with pytest.raises(ValueError, match="must all be positive"):
            compute_webster_green_splits(flow_ratios=[0.3, 0.0])

    def test_rejects_oversaturated_intersection(self) -> None:
        with pytest.raises(ValueError, match="oversaturated"):
            compute_webster_green_splits(flow_ratios=[0.6, 0.5])  # Y = 1.1 >= 1.0


class TestSelectFixedTimePhase:
    PHASES = (0, 2)
    DURATIONS = (30.0, 20.0)  # total cycle 50s

    def test_selects_first_phase_at_cycle_start(self) -> None:
        assert select_fixed_time_phase(self.PHASES, self.DURATIONS, 0.0) == 0

    def test_selects_first_phase_just_before_its_end(self) -> None:
        assert select_fixed_time_phase(self.PHASES, self.DURATIONS, 29.9) == 0

    def test_selects_second_phase_just_after_first_ends(self) -> None:
        assert select_fixed_time_phase(self.PHASES, self.DURATIONS, 30.1) == 2

    def test_wraps_around_to_first_phase_next_cycle(self) -> None:
        assert select_fixed_time_phase(self.PHASES, self.DURATIONS, 50.0 + 5.0) == 0

    def test_rejects_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            select_fixed_time_phase((0, 1, 2), (30.0, 20.0), 0.0)

    def test_rejects_zero_total_cycle(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            select_fixed_time_phase((0,), (0.0,), 0.0)


class TestNativeGreenDurationsFromNetFile:
    @classmethod
    @pytest.fixture(scope="class")
    def net_file(cls) -> Path:
        sys.path.insert(0, str(project_root() / "scripts"))
        import generate_sandbox_network

        generate_sandbox_network.generate()
        return generate_sandbox_network.NET_FILE

    def test_reads_b1_native_durations(self, net_file: Path) -> None:
        # B1's tlLogic (verified against net.net.xml before writing this): both green
        # phases (0 and 2) have duration="42".
        green_phases, durations = native_green_durations_from_net_file("B1", net_file)

        assert green_phases == (0, 2)
        assert durations == (42.0, 42.0)

    def test_raises_for_unknown_tls_id(self, net_file: Path) -> None:
        with pytest.raises(ValueError, match="No tlLogic found"):
            native_green_durations_from_net_file("not_a_real_junction", net_file)
