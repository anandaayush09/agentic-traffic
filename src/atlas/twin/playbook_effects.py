"""Compiles a playbook proposal into concrete changes a running twin can apply and later
revert -- the missing piece between "the Supervisor approved a playbook"
(`atlas.cognitive`) and "something actually changes on the road" (docs/design/
04_agent_architecture.md section 4.4: "playbooks compile to mode/limit changes").

Only playbooks with a defensible physical lever in this twin compile; the rest return
`None`, and callers (the what-if tool, the live twin) treat that as honestly
"unverifiable / not applied", never as a made-up effect:

| playbook                | effect in the twin                                              |
|-------------------------|-----------------------------------------------------------------|
| switch_to_max_pressure  | those junctions run the max-pressure controller                 |
| divert_arterial         | the road is penalised for routing; vehicles ahead of it reroute |
| meter_upstream          | upstream controlled junctions run max-pressure, which throttles |
|                         | inflow toward links whose downstream queues are full. `rate` is |
|                         | NOT enforced as a numeric admission rate -- a named limitation. |
| hold_current_plan       | nothing (explicitly a no-op)                                    |
| weather_capacity_derate | no compile: derating *assumed* capacity has no physical lever   |
| bus_bridge              | no compile: the network has no transit                          |

Takes plain `(name, params)` rather than `atlas.cognitive`'s proposal model: that package
already depends on `atlas.twin`, so importing it here would be a cycle.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass

from atlas.twin.incident_effects import RoadIndex

DEFAULT_EFFECT_DURATION_S = 900.0

COMPILABLE_PLAYBOOKS = frozenset(
    {"switch_to_max_pressure", "divert_arterial", "meter_upstream", "hold_current_plan"}
)


@dataclass(frozen=True, slots=True)
class PlaybookEffect:
    playbook_name: str
    summary: str
    junction_modes: Mapping[str, str]
    """junction id -> controller mode forced while the effect is active."""
    avoid_edges: tuple[str, ...]
    """roads penalised for routing while the effect is active."""
    duration_s: float
    affected_edges: tuple[str, ...]
    """roads whose traffic the effect is meant to relieve (impact is measured here)."""

    @property
    def is_noop(self) -> bool:
        return not self.junction_modes and not self.avoid_edges


def _str_list(value: object) -> list[str] | None:
    if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
        return [str(v) for v in value]
    return None


def compile_playbook(
    playbook_name: str,
    params: Mapping[str, object],
    index: RoadIndex,
    controlled_tls: Collection[str],
    default_duration_s: float = DEFAULT_EFFECT_DURATION_S,
) -> PlaybookEffect | None:
    """The effect of `playbook_name(params)`, or None if it can't be modelled (unknown /
    unsupported playbook, or its location doesn't resolve to anything this twin
    controls)."""
    if playbook_name == "hold_current_plan":
        return PlaybookEffect("hold_current_plan", "No change", {}, (), 0.0, ())

    if playbook_name == "switch_to_max_pressure":
        junction_ids = _str_list(params.get("junction_ids"))
        if junction_ids is None:
            return None
        controlled = [j for j in junction_ids if j in controlled_tls]
        if not controlled:
            return None
        affected = tuple(e for j in controlled for e in index.incoming.get(j, ()))
        return PlaybookEffect(
            "switch_to_max_pressure",
            f"Max-pressure control at {', '.join(controlled)}",
            {j: "max_pressure" for j in controlled},
            (),
            default_duration_s,
            affected,
        )

    if playbook_name == "divert_arterial":
        from_edge = params.get("from_edge")
        if not isinstance(from_edge, str):
            return None
        edges = index.resolve([from_edge])
        if not edges:
            return None
        duration = params.get("duration_s")
        return PlaybookEffect(
            "divert_arterial",
            f"Divert traffic away from {', '.join(edges)}",
            {},
            tuple(edges),
            float(duration)
            if isinstance(duration, int | float) and duration > 0
            else default_duration_s,
            tuple(edges),
        )

    if playbook_name == "meter_upstream":
        corridor_id = params.get("corridor_id")
        if not isinstance(corridor_id, str):
            return None
        edges = index.resolve([corridor_id])
        upstream: dict[str, None] = {}
        for edge_id in edges:
            near = index.from_node.get(edge_id)
            if near is None:
                continue
            candidates = {near} | {
                index.from_node[p] for p in index.incoming.get(near, ()) if p in index.from_node
            }
            for junction in sorted(candidates):
                if junction in controlled_tls:
                    upstream.setdefault(junction, None)
        if not upstream:
            return None
        return PlaybookEffect(
            "meter_upstream",
            f"Meter inflow via max-pressure at {', '.join(upstream)}",
            {j: "max_pressure" for j in upstream},
            (),
            default_duration_s,
            tuple(edges),
        )

    return None
