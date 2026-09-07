"""Hand-computed verification of green-wave offset math and spillback detection --
per this project's established standard (max-pressure, Webster, GAE, PPO clipping were
all verified this way, not just "does it run").
"""

from __future__ import annotations

import pytest

from atlas.coordination.greenwave import compute_greenwave_offsets, detect_spillback


class TestComputeGreenwaveOffsets:
    def test_hand_computed_three_junctions(self) -> None:
        """distances=[100, 200]m, speed=10 m/s:
        offset[0] = 0
        offset[1] = 100/10 = 10
        offset[2] = 10 + 200/10 = 30
        """
        offsets = compute_greenwave_offsets([100.0, 200.0], platoon_speed_mps=10.0)
        assert offsets == pytest.approx((0.0, 10.0, 30.0))

    def test_single_junction_no_segments_returns_only_zero_offset(self) -> None:
        assert compute_greenwave_offsets([], platoon_speed_mps=10.0) == (0.0,)

    def test_zero_distance_segment_gives_zero_additional_offset(self) -> None:
        offsets = compute_greenwave_offsets([0.0, 50.0], platoon_speed_mps=5.0)
        assert offsets == pytest.approx((0.0, 0.0, 10.0))

    def test_rejects_non_positive_speed(self) -> None:
        with pytest.raises(ValueError, match="platoon_speed_mps"):
            compute_greenwave_offsets([100.0], platoon_speed_mps=0.0)

    def test_rejects_negative_distance(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            compute_greenwave_offsets([-10.0], platoon_speed_mps=10.0)

    def test_slower_platoon_speed_gives_larger_offsets(self) -> None:
        fast = compute_greenwave_offsets([100.0], platoon_speed_mps=20.0)
        slow = compute_greenwave_offsets([100.0], platoon_speed_mps=10.0)
        assert slow[1] > fast[1]


class TestDetectSpillback:
    def test_above_threshold_is_spillback(self) -> None:
        assert detect_spillback(0.9, threshold=0.85) is True

    def test_exactly_at_threshold_is_spillback(self) -> None:
        assert detect_spillback(0.85, threshold=0.85) is True

    def test_below_threshold_is_not_spillback(self) -> None:
        assert detect_spillback(0.5, threshold=0.85) is False

    def test_rejects_occupancy_out_of_range(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            detect_spillback(1.5)
