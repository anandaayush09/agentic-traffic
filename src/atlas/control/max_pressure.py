"""Max-pressure signal control (Varaiya 2013, via the backpressure/max-weight scheduling
literature): choose the green phase maximizing total pressure, where pressure for one
movement is (queue on its upstream lane - queue on its downstream lane), summed over
every movement the phase enables. Provably stability-maximizing under standard
assumptions, fully decentralized, model-free -- see docs/design/02_literature_survey.md
section 2.6. Serves two roles in the frozen architecture: a strong classical baseline
(Part 11 evaluation) and the safety shield's degradation fallback (RL -> max-pressure ->
fixed-time, Part 3.10).

Movement structure comes directly from the built network's `<connection tl=... link
Index=...>` elements (verified against real data before implementing): `linkIndex` is the
character position in that junction's tlLogic phase-state strings, so connecting
linkIndex -> (from_lane, to_lane) plus each phase's state string is enough to know exactly
which movements are enabled ("G"/"g") in each phase -- no separate geometry reasoning
needed. Note a single physical lane can appear as the "from_lane" of multiple movements
(e.g. a shared through+right lane gets two linkIndexes); pressure is summed per movement,
not deduplicated per lane, matching the standard formulation (and PressLight's phase-
pressure reward, which this project's control-plane reward is also built on).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

from atlas.control.phases import green_phase_indices as _green_phase_indices


@dataclass(frozen=True, slots=True)
class Movement:
    """One controllable turning movement: a single signal-group position controlling
    traffic from one lane to another."""

    link_index: int
    from_lane: str
    to_lane: str


@dataclass(frozen=True, slots=True)
class MaxPressureProgram:
    """The movement structure and phase definitions needed to compute max-pressure for
    one junction."""

    tls_id: str
    movements: tuple[Movement, ...]
    green_phase_indices: tuple[int, ...]
    phase_states: tuple[str, ...]
    """Every phase's state string (yellow phases included), indexed by SUMO phase index."""

    @classmethod
    def from_net_file(cls, tls_id: str, net_file: Path) -> MaxPressureProgram:
        root = ElementTree.parse(net_file).getroot()

        movements = tuple(
            Movement(
                link_index=int(conn.get("linkIndex", "-1")),
                from_lane=f"{conn.get('from')}_{conn.get('fromLane')}",
                to_lane=f"{conn.get('to')}_{conn.get('toLane')}",
            )
            for conn in root.findall("connection")
            if conn.get("tl") == tls_id
        )
        if not movements:
            raise ValueError(f"No connections found for traffic light '{tls_id}' in {net_file}")

        logic = root.find(f".//tlLogic[@id='{tls_id}']")
        if logic is None:
            raise ValueError(f"No tlLogic found for traffic light '{tls_id}' in {net_file}")
        phase_states = tuple(phase.get("state", "") for phase in logic.findall("phase"))
        green = _green_phase_indices(phase_states)
        if not green:
            raise ValueError(f"No green phases found for traffic light '{tls_id}'")

        return cls(
            tls_id=tls_id,
            movements=movements,
            green_phase_indices=green,
            phase_states=phase_states,
        )

    def enabled_movements(self, phase_index: int) -> tuple[Movement, ...]:
        """Movements enabled ('G' or 'g') by the given phase."""
        state = self.phase_states[phase_index]
        return tuple(m for m in self.movements if state[m.link_index] in "Gg")


def phase_pressure(
    program: MaxPressureProgram, phase_index: int, queue_lengths: Mapping[str, int]
) -> int:
    """Total pressure for one phase: sum over its enabled movements of
    (upstream queue - downstream queue). `queue_lengths` maps lane_id -> halting vehicle
    count (LaneState.queue_length); a lane missing from the mapping is treated as empty.
    """
    return sum(
        queue_lengths.get(m.from_lane, 0) - queue_lengths.get(m.to_lane, 0)
        for m in program.enabled_movements(phase_index)
    )


def total_pressure(program: MaxPressureProgram, queue_lengths: Mapping[str, int]) -> int:
    """Total pressure summed over EVERY movement the junction controls, regardless of
    which phase is currently active -- a whole-junction congestion measure, distinct from
    `phase_pressure`'s per-phase score used by the control law. This is the PressLight-
    style reward signal (docs/design/06_ai_pipeline.md section 6.1: "reward: negative
    junction pressure"): minimizing it (maximizing its negation) encourages keeping every
    approach's queues low, not just the ones currently being served.
    """
    return sum(
        queue_lengths.get(m.from_lane, 0) - queue_lengths.get(m.to_lane, 0)
        for m in program.movements
    )


def select_max_pressure_phase(program: MaxPressureProgram, queue_lengths: Mapping[str, int]) -> int:
    """The max-pressure control law: the green phase with the highest total pressure.
    Ties broken by lowest phase index, for determinism.
    """
    best_phase = program.green_phase_indices[0]
    best_pressure: int | None = None
    for phase in program.green_phase_indices:
        pressure = phase_pressure(program, phase, queue_lengths)
        if best_pressure is None or pressure > best_pressure:
            best_pressure = pressure
            best_phase = phase
    return best_phase
