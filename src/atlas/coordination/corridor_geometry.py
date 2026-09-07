"""Reads real inter-junction distances from the built network -- the missing input
`greenwave.compute_greenwave_offsets` needs before it can run against a live corridor
(that function takes distances as a plain argument, deliberately not sourcing them
itself, per its own module docstring). Verified against manhattan_midtown's real
corridor (42430333 through 42430347, the same 5 junctions Week 5's corridor training
uses) before writing this: every consecutive pair has exactly one direct edge between
them (63-67m each), not a multi-hop path -- so a direct from-node/to-node edge lookup is
sufficient for this network, with a straight-line fallback documented for the general
case where it isn't.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import sumolib


def compute_corridor_segment_distances_m(net_file: Path, tls_ids: Sequence[str]) -> list[float]:
    """One distance per consecutive pair in `tls_ids` (length `len(tls_ids) - 1`), ready
    to pass directly to `greenwave.compute_greenwave_offsets`. Prefers a direct edge's
    real length (either direction, since arterials are often one-way); falls back to
    straight-line node-to-node distance if no direct edge connects the pair -- a real
    but named approximation, not silently assumed exact, for the fewer-common case of a
    corridor with an intermediate unsignalized junction between two signalized ones.
    """
    if len(tls_ids) < 2:
        raise ValueError("tls_ids must have at least 2 junctions to measure a segment")

    net = sumolib.net.readNet(str(net_file))
    distances: list[float] = []
    for tls_a, tls_b in zip(tls_ids, tls_ids[1:], strict=False):
        distances.append(_segment_distance_m(net, tls_a, tls_b))
    return distances


def _segment_distance_m(net: sumolib.net.Net, tls_a: str, tls_b: str) -> float:
    direct_edge = _direct_edge(net, tls_a, tls_b) or _direct_edge(net, tls_b, tls_a)
    if direct_edge is not None:
        return float(direct_edge.getLength())

    node_a, node_b = net.getNode(tls_a), net.getNode(tls_b)
    (xa, ya), (xb, yb) = node_a.getCoord(), node_b.getCoord()
    return math.dist((xa, ya), (xb, yb))


def _direct_edge(net: sumolib.net.Net, from_tls: str, to_tls: str) -> sumolib.net.edge.Edge | None:
    for edge in net.getEdges():
        if edge.getFromNode().getID() == from_tls and edge.getToNode().getID() == to_tls:
            return edge
    return None
