"""Hand-computed verification of virtual-loop counting: line crossing and occupancy
geometry, independent of any real detector/tracker (no GPU/YOLO/UA-DETRAC needed).
"""

from __future__ import annotations

import pytest

from atlas.perception.virtual_loop import LineCrossingCounter, VirtualLoop, occupancy_fraction

# Horizontal loop line at y=0, from x=-10 to x=10.
HORIZONTAL_LOOP = VirtualLoop(p1=(-10.0, 0.0), p2=(10.0, 0.0))


class TestLineCrossingCounter:
    def test_first_update_never_counts_as_a_crossing(self) -> None:
        counter = LineCrossingCounter(HORIZONTAL_LOOP)

        crossed = counter.update("car_1", (0.0, 5.0))

        assert crossed is False
        assert counter.count == 0

    def test_moving_from_one_side_to_the_other_counts_as_one_crossing(self) -> None:
        counter = LineCrossingCounter(HORIZONTAL_LOOP)
        counter.update("car_1", (0.0, -5.0))  # below the line

        crossed = counter.update("car_1", (0.0, 5.0))  # now above

        assert crossed is True
        assert counter.count == 1

    def test_staying_on_the_same_side_does_not_count(self) -> None:
        counter = LineCrossingCounter(HORIZONTAL_LOOP)
        counter.update("car_1", (0.0, -5.0))

        crossed = counter.update("car_1", (0.0, -1.0))  # still below

        assert crossed is False
        assert counter.count == 0

    def test_multiple_tracks_counted_independently(self) -> None:
        counter = LineCrossingCounter(HORIZONTAL_LOOP)
        counter.update("car_1", (0.0, -5.0))
        counter.update("car_2", (5.0, -3.0))

        counter.update("car_1", (0.0, 5.0))  # car_1 crosses
        counter.update("car_2", (5.0, -1.0))  # car_2 does not

        assert counter.count == 1

    def test_reset_count_zeros_the_running_total_but_keeps_track_state(self) -> None:
        counter = LineCrossingCounter(HORIZONTAL_LOOP)
        counter.update("car_1", (0.0, -5.0))
        counter.update("car_1", (0.0, 5.0))  # crosses, count=1
        counter.reset_count()

        assert counter.count == 0

        # car_1 is still tracked as being above the line -- moving back below is a
        # real second crossing, not silently ignored just because we reset the count.
        crossed = counter.update("car_1", (0.0, -5.0))
        assert crossed is True
        assert counter.count == 1

    def test_sitting_exactly_on_the_line_does_not_register_as_a_crossing(self) -> None:
        counter = LineCrossingCounter(HORIZONTAL_LOOP)
        counter.update("car_1", (0.0, -5.0))

        crossed = counter.update("car_1", (0.0, 0.0))  # exactly on the line

        assert crossed is False


class TestOccupancyFraction:
    def test_empty_positions_is_zero_occupancy(self) -> None:
        assert occupancy_fraction(HORIZONTAL_LOOP, [], proximity_m=1.0) == 0.0

    def test_all_positions_near_the_line_is_full_occupancy(self) -> None:
        positions = [(0.0, 0.5), (2.0, -0.5), (-3.0, 0.0)]
        assert occupancy_fraction(HORIZONTAL_LOOP, positions, proximity_m=1.0) == 1.0

    def test_half_near_half_far_gives_half_occupancy(self) -> None:
        positions = [(0.0, 0.0), (0.0, 100.0)]  # one on the line, one far away
        assert occupancy_fraction(HORIZONTAL_LOOP, positions, proximity_m=1.0) == pytest.approx(0.5)

    def test_position_beyond_the_segment_endpoint_uses_endpoint_distance(self) -> None:
        # x=20 is past the loop's right endpoint (x=10) -- nearest point on the
        # SEGMENT is the endpoint (10, 0), distance = 10, not the infinite-line
        # distance (which would be 0 since y matches).
        positions = [(20.0, 0.0)]
        assert occupancy_fraction(HORIZONTAL_LOOP, positions, proximity_m=5.0) == 0.0
        assert occupancy_fraction(HORIZONTAL_LOOP, positions, proximity_m=15.0) == 1.0
