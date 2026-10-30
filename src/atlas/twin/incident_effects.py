"""Pure incident -> road-disruption mapping, shared by everything that needs to put an
incident into a running twin: the live-traffic service (the dashboard map) and the
what-if tool (simulating an incident with and without a playbook response).

Kept free of any simulation, service or cognitive-plane imports so both can depend on it
without creating an import cycle (`atlas.cognitive` already depends on `atlas.twin`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from atlas.twin.scenario import ScenarioSpec

# Incident classes that have a physical effect on a road. Everything else (sensor_fault,
# comms_fault, event_surge, emergency_run, ...) is a no-op for the twin.
ROAD_EFFECT_CLASSES = frozenset({"accident", "closure", "construction", "weather"})

_ACCIDENT_SPEED_FACTOR = 0.25
_WEATHER_SPEED_FACTOR = 0.6
_CONSTRUCTION_SPEED_FACTOR = 0.7
_SINGLE_LANE_CONSTRUCTION_SPEED_FACTOR = 0.5


@dataclass(frozen=True, slots=True)
class RoadIndex:
    """The static road-network facts incident/playbook handling needs, read once from a
    built SUMO network (no simulation required)."""

    lane_counts: Mapping[str, int]
    free_flow_mps: Mapping[str, float]
    length_m: Mapping[str, float]
    from_node: Mapping[str, str]
    to_node: Mapping[str, str]
    incoming: Mapping[str, tuple[str, ...]]
    """node id -> ids of the roads entering it."""

    @classmethod
    def from_net(cls, net: Any) -> RoadIndex:
        edges = [e for e in net.getEdges() if not e.getID().startswith(":")]
        incoming: dict[str, tuple[str, ...]] = {}
        for node in net.getNodes():
            incoming[node.getID()] = tuple(
                e.getID() for e in node.getIncoming() if not e.getID().startswith(":")
            )
        return cls(
            lane_counts={e.getID(): int(e.getLaneNumber()) for e in edges},
            free_flow_mps={e.getID(): float(e.getSpeed()) for e in edges},
            length_m={e.getID(): float(e.getLength()) for e in edges},
            from_node={e.getID(): e.getFromNode().getID() for e in edges},
            to_node={e.getID(): e.getToNode().getID() for e in edges},
            incoming=incoming,
        )

    @classmethod
    def from_net_file(cls, net_file: Path) -> RoadIndex:
        import libsumo  # noqa: F401  -- import before sumolib/pyproj, established DLL-order fix
        import sumolib

        return cls.from_net(sumolib.net.readNet(str(net_file)))

    def resolve(self, link_ids: Sequence[str]) -> list[str]:
        """Road ids for an incident/playbook location. Each id may be a road (that road)
        or a junction (every road entering it) -- the dashboard keys incidents by
        junction. Unknown ids are skipped; order is preserved, duplicates dropped."""
        resolved: dict[str, None] = {}
        for link_id in link_ids:
            if link_id in self.lane_counts:
                resolved.setdefault(link_id, None)
            else:
                for edge_id in self.incoming.get(link_id, ()):
                    resolved.setdefault(edge_id, None)
        return list(resolved)


def link_ids_of(location: object) -> list[str]:
    """`location.link_ids` of an incident as strings (empty if absent/malformed)."""
    if isinstance(location, Mapping):
        raw = location.get("link_ids")
        if isinstance(raw, list):
            return [str(x) for x in raw]
    return []


def plan_disruption_specs(
    incident_id: str,
    incident_class: str,
    edge_lane_counts: Mapping[str, int],
    start_s: int,
    duration_s: int,
) -> list[ScenarioSpec]:
    """One scenario-engine spec per affected road (so each can be tracked and reverted
    independently). Same severities the committed scenario config
    (`configs/scenarios/manhattan_midtown.yaml`) uses for its first variant per class.
    """
    specs: list[ScenarioSpec] = []
    for edge_id, lane_count in edge_lane_counts.items():
        name = f"{incident_id}:{edge_id}"
        common: dict[str, Any] = {
            "name": name,
            "edge_id": edge_id,
            "start_s": start_s,
            "duration_s": duration_s,
        }
        if incident_class == "accident":
            specs.append(
                ScenarioSpec(type="accident", speed_factor=_ACCIDENT_SPEED_FACTOR, **common)
            )
        elif incident_class == "closure":
            specs.append(ScenarioSpec(type="closure", **common))
        elif incident_class == "weather":
            specs.append(ScenarioSpec(type="weather", speed_factor=_WEATHER_SPEED_FACTOR, **common))
        elif incident_class == "construction":
            if lane_count >= 2:
                # A real lane drop: block the first lane, derate the rest.
                specs.append(
                    ScenarioSpec(
                        type="construction",
                        speed_factor=_CONSTRUCTION_SPEED_FACTOR,
                        lane_indices=[0],
                        **common,
                    )
                )
            else:
                # A single-lane edge has no lane to drop without closing it outright.
                specs.append(
                    ScenarioSpec(
                        type="accident",
                        speed_factor=_SINGLE_LANE_CONSTRUCTION_SPEED_FACTOR,
                        **common,
                    )
                )
    return specs
