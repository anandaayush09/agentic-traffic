"""Hand-computed verification of fit_error_model's per-condition statistics."""

from __future__ import annotations

import pytest

from atlas.perception.error_model import DetectionTally, fit_error_model


class TestFitErrorModel:
    def test_hand_computed_single_condition(self) -> None:
        """Two clips, condition="day":
        clip 1: tp=8, fp=2, fn=2, gt=10, detected=10
        clip 2: tp=6, fp=1, fn=4, gt=10, detected=7

        miss_rate = (2+4) / (8+6+2+4) = 6/20 = 0.30
        fp_rate   = (2+1) / (8+6+2+1) = 3/17 ~= 0.1765
        count_mae = (|10-10| + |7-10|) / 2 = (0 + 3) / 2 = 1.5
        """
        tallies = [
            DetectionTally(
                "day",
                ground_truth_count=10,
                detected_count=10,
                true_positives=8,
                false_positives=2,
                false_negatives=2,
            ),
            DetectionTally(
                "day",
                ground_truth_count=10,
                detected_count=7,
                true_positives=6,
                false_positives=1,
                false_negatives=4,
            ),
        ]

        stats = fit_error_model(tallies)

        assert stats["day"].miss_rate == pytest.approx(0.30)
        assert stats["day"].fp_rate == pytest.approx(3 / 17)
        assert stats["day"].count_mae == pytest.approx(1.5)
        assert stats["day"].num_clips == 2

    def test_groups_by_condition_independently(self) -> None:
        tallies = [
            DetectionTally("day", 10, 10, true_positives=10, false_positives=0, false_negatives=0),
            DetectionTally("night", 10, 5, true_positives=5, false_positives=0, false_negatives=5),
        ]

        stats = fit_error_model(tallies)

        assert stats["day"].miss_rate == pytest.approx(0.0)
        assert stats["night"].miss_rate == pytest.approx(0.5)
        assert set(stats.keys()) == {"day", "night"}

    def test_perfect_detection_has_zero_error(self) -> None:
        tallies = [
            DetectionTally("day", 5, 5, true_positives=5, false_positives=0, false_negatives=0)
        ]

        stats = fit_error_model(tallies)

        assert stats["day"].miss_rate == 0.0
        assert stats["day"].fp_rate == 0.0
        assert stats["day"].count_mae == 0.0

    def test_no_true_positives_or_false_negatives_gives_zero_miss_rate_not_nan(self) -> None:
        """Degenerate case: a clip with zero ground-truth vehicles at all -- must not
        divide by zero."""
        tallies = [
            DetectionTally("empty", 0, 0, true_positives=0, false_positives=0, false_negatives=0)
        ]

        stats = fit_error_model(tallies)

        assert stats["empty"].miss_rate == 0.0
        assert stats["empty"].fp_rate == 0.0

    def test_rejects_empty_input(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            fit_error_model([])
