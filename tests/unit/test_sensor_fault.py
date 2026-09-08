"""Validates SensorFaultModel against deterministic edge cases (0%/100% dropout) and
a seeded-RNG statistical check at a real fractional rate -- not just "the type
signature looks right."
"""

from __future__ import annotations

import random

import pytest

from atlas.twin.sensor_fault import SensorFaultModel

LANES = {f"lane_{i}": 5 for i in range(1000)}


class TestConstruction:
    def test_rejects_out_of_range_dropout_rate(self) -> None:
        with pytest.raises(ValueError, match="dropout_rate must be in"):
            SensorFaultModel(dropout_rate=1.5, rng=random.Random(0))

    def test_rejects_negative_dropout_rate(self) -> None:
        with pytest.raises(ValueError, match="dropout_rate must be in"):
            SensorFaultModel(dropout_rate=-0.1, rng=random.Random(0))


class TestApply:
    def test_zero_dropout_never_drops_anything(self) -> None:
        model = SensorFaultModel(dropout_rate=0.0, rng=random.Random(42))

        result = model.apply(LANES)

        assert result == LANES

    def test_full_dropout_drops_everything(self) -> None:
        model = SensorFaultModel(dropout_rate=1.0, rng=random.Random(42))

        result = model.apply(LANES)

        assert result == {}

    def test_missing_lanes_are_removed_not_zeroed(self) -> None:
        """A dropped-out lane must be ABSENT from the result, not present with value
        0 -- max_pressure.py's own convention treats "missing" as "unknown, assume
        empty" and "present with 0" as "confirmed empty," a real difference."""
        model = SensorFaultModel(dropout_rate=1.0, rng=random.Random(42))

        result = model.apply({"lane_a": 5})

        assert "lane_a" not in result
        assert "lane_a" not in result  # explicit: not result.get("lane_a") == 0

    def test_partial_dropout_rate_matches_expectation_statistically(self) -> None:
        """1000 lanes, dropout_rate=0.3, seeded RNG -- the observed drop fraction
        should land close to 30% (a real statistical check, not just "some dropped").
        """
        model = SensorFaultModel(dropout_rate=0.3, rng=random.Random(1234))

        result = model.apply(LANES)

        observed_drop_fraction = 1 - (len(result) / len(LANES))
        assert observed_drop_fraction == pytest.approx(0.3, abs=0.03)

    def test_independent_calls_can_drop_different_lanes(self) -> None:
        model = SensorFaultModel(dropout_rate=0.3, rng=random.Random(7))

        first = model.apply(LANES)
        second = model.apply(LANES)

        assert first != second  # a lane down on one reading need not be down on the next


class TestFromErrorModelStats:
    def test_miss_rate_becomes_dropout_rate_directly(self) -> None:
        model = SensorFaultModel.from_error_model_stats(miss_rate=0.12, rng=random.Random(0))

        assert model.dropout_rate == 0.12
