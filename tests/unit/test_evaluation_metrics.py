"""Validates atlas.evaluation.metrics against hand-computed values and a real SUMO
tripinfo file (parsed exactly as scripts/eval-style scripts would encounter it) --
same discipline as every other math module in this project.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from atlas.evaluation.metrics import (
    TripRecord,
    _percentile,
    compute_traffic_metrics,
    parse_tripinfo,
)

REAL_TRIPINFO_XML = """<?xml version="1.0" encoding="UTF-8"?>
<tripinfos>
    <tripinfo id="v1" depart="2.00" arrival="29.00" duration="27.00" waitingTime="3.00"/>
    <tripinfo id="v2" depart="5.00" arrival="40.00" duration="35.00" waitingTime="10.00"/>
    <tripinfo id="v3" depart="90.00" arrival="-1" duration="10.00" waitingTime="1.00"/>
</tripinfos>
"""


class TestParseTripinfo:
    def test_parses_a_real_tripinfo_file_including_an_unfinished_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "tripinfo.xml"
        path.write_text(REAL_TRIPINFO_XML)

        trips = parse_tripinfo(path)

        assert len(trips) == 3
        by_id = {t.vehicle_id: t for t in trips}
        assert by_id["v1"].duration_s == 27.0
        assert by_id["v1"].waiting_time_s == 3.0
        assert by_id["v1"].completed is True
        assert by_id["v3"].completed is False  # arrival="-1" -- still en route at sim end


class TestPercentile:
    def test_matches_hand_computed_linear_interpolation(self) -> None:
        """[1,2,3,4,5], p=0.95: rank = 0.95*4 = 3.8 -> between index 3 (value 4) and
        index 4 (value 5), fraction 0.8 -> 4 + (5-4)*0.8 = 4.8."""
        result = _percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.95)

        assert result == pytest.approx(4.8)

    def test_p50_of_an_odd_length_list_is_the_exact_middle_value(self) -> None:
        result = _percentile([5.0, 1.0, 3.0, 2.0, 4.0], 0.5)  # unsorted input on purpose

        assert result == pytest.approx(3.0)

    def test_p0_and_p1_are_the_min_and_max(self) -> None:
        values = [7.0, 2.0, 9.0, 4.0]

        assert _percentile(values, 0.0) == 2.0
        assert _percentile(values, 1.0) == 9.0

    def test_single_value_list(self) -> None:
        assert _percentile([42.0], 0.95) == 42.0

    def test_raises_on_empty_list(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            _percentile([], 0.5)


class TestComputeTrafficMetrics:
    def test_matches_hand_computed_values(self) -> None:
        """3 trips (2 completed, 1 not): mean_waiting = (3+10+1)/3 = 4.6667,
        mean_travel = (27+35+10)/3 = 24.0, throughput = 2 completed / (3600s/3600) =
        2.0 veh/hour for a 1-hour episode. queue_samples=[0,2,4,6,8]: mean=4.0,
        p95 (rank=0.95*4=3.8, between 6 and 8, frac 0.8) = 6 + 2*0.8 = 7.6.
        """
        trips = [
            TripRecord(vehicle_id="v1", duration_s=27.0, waiting_time_s=3.0, completed=True),
            TripRecord(vehicle_id="v2", duration_s=35.0, waiting_time_s=10.0, completed=True),
            TripRecord(vehicle_id="v3", duration_s=10.0, waiting_time_s=1.0, completed=False),
        ]
        queue_samples = [0.0, 2.0, 4.0, 6.0, 8.0]

        metrics = compute_traffic_metrics(trips, queue_samples, episode_duration_s=3600.0)

        assert metrics.mean_waiting_time_s == pytest.approx(4.6667, abs=1e-3)
        assert metrics.mean_travel_time_s == pytest.approx(24.0)
        assert metrics.throughput_veh_per_hour == pytest.approx(2.0)
        assert metrics.mean_queue_length == pytest.approx(4.0)
        assert metrics.p95_queue_length == pytest.approx(7.6)
        assert metrics.num_trips == 3
        assert metrics.num_completed_trips == 2

    def test_raises_on_zero_trips(self) -> None:
        with pytest.raises(ValueError, match="zero trips"):
            compute_traffic_metrics([], [1.0], episode_duration_s=100.0)

    def test_raises_on_zero_queue_samples(self) -> None:
        trip = TripRecord(vehicle_id="v1", duration_s=1.0, waiting_time_s=0.0, completed=True)
        with pytest.raises(ValueError, match="zero samples"):
            compute_traffic_metrics([trip], [], episode_duration_s=100.0)
