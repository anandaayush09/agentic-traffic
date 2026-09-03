"""Demand configuration and turn-direction auditing.

Two concerns live here, both feeding Week 2's "plausibility invariant" validation
strategy (docs/design/08_data_strategy.md section 8.2): since no real traffic-count
ground truth exists for a synthetic district, demand is validated against invariants
(a congestion proxy, turn-direction sanity) rather than fabricated "real" numbers.

- `DemandPeriod`/`DemandConfig`: declarative schema for configs/demand/*.yaml.
- `parse_connection_directions`/`turn_direction_counts`: read the turn direction SUMO's
  own `netconvert` already computed for each network connection (the `dir` attribute:
  straight/left/right/turn), and tally how a generated route file's trips use them. This
  is the audit half of "turn-ratio calibration" -- the shaping half is
  `randomTrips.py --fringe-factor`, applied in scripts/generate_demand.py.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from xml.etree import ElementTree

from pydantic import BaseModel, Field


class DemandPeriod(BaseModel):
    """One named demand period (e.g. "am_peak"): a randomTrips.py generation recipe."""

    name: str
    duration_s: int = Field(gt=0)
    period_s: float = Field(gt=0)
    """Mean seconds between successive vehicle insertions (randomTrips.py's -p)."""
    fringe_factor: float = Field(ge=0)
    """Bias toward longer fringe-to-fringe trips over short internal hops (-\\-fringe-factor)."""
    seed: int


class DemandConfig(BaseModel):
    network: str
    periods: list[DemandPeriod]


def parse_connection_directions(net_file: Path) -> dict[tuple[str, str], str]:
    """Map (from_edge, to_edge) -> SUMO's computed turn direction code ('s'/'l'/'r'/'t'/...)
    by reading <connection> elements out of a built net.net.xml. Multiple lane-level
    connections can share the same edge pair; they always agree on direction, so a dict
    (last write wins) is sufficient.
    """
    directions: dict[tuple[str, str], str] = {}
    for _, elem in ElementTree.iterparse(net_file, events=("end",)):
        if elem.tag != "connection":
            continue
        from_edge = elem.get("from")
        to_edge = elem.get("to")
        direction = elem.get("dir")
        if from_edge and to_edge and direction:
            directions[(from_edge, to_edge)] = direction
        elem.clear()
    return directions


def turn_direction_counts(
    route_file: Path, connection_dirs: dict[tuple[str, str], str]
) -> Counter[str]:
    """Tally how many consecutive-edge transitions in a route file's vehicle routes fall
    into each turn-direction category. Transitions that cross an edge pair with no known
    connection (e.g. via an internal junction not captured verbatim) are skipped rather
    than guessed at.
    """
    counts: Counter[str] = Counter()
    for _, elem in ElementTree.iterparse(route_file, events=("end",)):
        if elem.tag != "route":
            continue
        edges_attr = elem.get("edges")
        if not edges_attr:
            elem.clear()
            continue
        edges = edges_attr.split()
        for from_edge, to_edge in zip(edges, edges[1:], strict=False):
            direction = connection_dirs.get((from_edge, to_edge))
            if direction is not None:
                counts[direction] += 1
        elem.clear()
    return counts
