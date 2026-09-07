"""Validates compute_corridor_segment_distances_m against real network data: the direct-
edge path (manhattan_midtown's real corridor, exact lengths verified via sumolib before
writing this) and the straight-line fallback (sandbox's B0/B2, which aren't directly
connected -- both go through hub junction B1).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from atlas.common.config import project_root
from atlas.coordination.corridor_geometry import compute_corridor_segment_distances_m

MANHATTAN_NET = project_root() / "data" / "networks" / "manhattan_midtown" / "net.net.xml"
# The same 5 junctions Week 5's corridor training uses -- verified collinear, and each
# consecutive pair verified (via sumolib, before writing this test) to have exactly one
# direct edge with these lengths.
CORRIDOR_TLS_IDS = ["42430333", "42430338", "42430342", "42430344", "42430347"]
EXPECTED_DIRECT_EDGE_LENGTHS_M = [63.22, 65.12, 66.87, 64.38]


@pytest.fixture(scope="session")
def sandbox_net_file() -> Path:
    import sys

    sys.path.insert(0, str(project_root() / "scripts"))
    import generate_sandbox_network

    generate_sandbox_network.generate()
    return generate_sandbox_network.NET_FILE


class TestDirectEdgeDistances:
    def test_matches_real_edge_lengths_for_manhattan_corridor(self) -> None:
        distances = compute_corridor_segment_distances_m(MANHATTAN_NET, CORRIDOR_TLS_IDS)
        assert distances == pytest.approx(EXPECTED_DIRECT_EDGE_LENGTHS_M, abs=0.01)

    def test_returns_one_fewer_distance_than_junctions(self) -> None:
        distances = compute_corridor_segment_distances_m(MANHATTAN_NET, CORRIDOR_TLS_IDS)
        assert len(distances) == len(CORRIDOR_TLS_IDS) - 1

    def test_rejects_fewer_than_two_junctions(self) -> None:
        with pytest.raises(ValueError, match="at least 2 junctions"):
            compute_corridor_segment_distances_m(MANHATTAN_NET, ["42430333"])


class TestFallbackToStraightLineDistance:
    def test_non_adjacent_junctions_still_return_a_positive_distance(
        self, sandbox_net_file: Path
    ) -> None:
        """B0 and B2 aren't directly connected in the sandbox network (both route
        through hub junction B1) -- must fall back to straight-line distance rather
        than crash or silently return 0."""
        distances = compute_corridor_segment_distances_m(sandbox_net_file, ["B0", "B2"])
        assert len(distances) == 1
        assert distances[0] > 0

    def test_directly_connected_pair_uses_real_edge_length_not_straight_line(
        self, sandbox_net_file: Path
    ) -> None:
        distances = compute_corridor_segment_distances_m(sandbox_net_file, ["A1", "B1"])
        assert distances[0] == pytest.approx(85.6, abs=0.1)
