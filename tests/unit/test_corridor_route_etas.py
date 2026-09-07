"""Hand-computed verification of compute_route_etas_s against the real manhattan_midtown
corridor's edge lengths (already verified in test_corridor_geometry.py).
"""

from __future__ import annotations

import pytest

from atlas.common.config import project_root
from atlas.coordination.corridor_geometry import compute_route_etas_s

NET_FILE = project_root() / "data" / "networks" / "manhattan_midtown" / "net.net.xml"
CORRIDOR_ROUTE_EDGES = ["127693103#0", "127693103#3", "464516471#0", "464516471#3"]
EDGE_LENGTHS_M = [63.22, 65.12, 66.87, 64.38]


class TestComputeRouteEtasS:
    def test_hand_computed_cumulative_etas_at_10mps(self) -> None:
        etas = compute_route_etas_s(NET_FILE, CORRIDOR_ROUTE_EDGES, speed_mps=10.0)

        expected = [
            EDGE_LENGTHS_M[0] / 10.0,
            (EDGE_LENGTHS_M[0] + EDGE_LENGTHS_M[1]) / 10.0,
            (EDGE_LENGTHS_M[0] + EDGE_LENGTHS_M[1] + EDGE_LENGTHS_M[2]) / 10.0,
            sum(EDGE_LENGTHS_M) / 10.0,
        ]
        assert etas == pytest.approx(expected, abs=0.01)

    def test_returns_one_eta_per_edge(self) -> None:
        etas = compute_route_etas_s(NET_FILE, CORRIDOR_ROUTE_EDGES, speed_mps=10.0)
        assert len(etas) == len(CORRIDOR_ROUTE_EDGES)

    def test_etas_are_monotonically_increasing(self) -> None:
        etas = compute_route_etas_s(NET_FILE, CORRIDOR_ROUTE_EDGES, speed_mps=12.0)
        assert etas == sorted(etas)

    def test_faster_speed_gives_smaller_etas(self) -> None:
        slow = compute_route_etas_s(NET_FILE, CORRIDOR_ROUTE_EDGES, speed_mps=10.0)
        fast = compute_route_etas_s(NET_FILE, CORRIDOR_ROUTE_EDGES, speed_mps=20.0)
        assert fast[-1] < slow[-1]

    def test_rejects_non_positive_speed(self) -> None:
        with pytest.raises(ValueError, match="speed_mps"):
            compute_route_etas_s(NET_FILE, CORRIDOR_ROUTE_EDGES, speed_mps=0.0)
