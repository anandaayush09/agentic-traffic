"""Shared parsing of SUMO tlLogic phase state strings. Used by the safety shield and both
classical controllers, all of which need to distinguish controllable "green" phases from
executor-managed "yellow" transition phases -- see shield.py's module docstring for the
full rationale.
"""

from __future__ import annotations

from collections.abc import Sequence


def is_yellow_phase(phase_state: str) -> bool:
    return any(c in "yY" for c in phase_state)


def green_phase_indices(phase_states: Sequence[str]) -> tuple[int, ...]:
    """Indices of the non-yellow phases in `phase_states`, in their original order."""
    return tuple(i for i, state in enumerate(phase_states) if not is_yellow_phase(state))


def yellow_phase_after(phase_states: Sequence[str], green_phase: int) -> int:
    """The yellow transition phase immediately following `green_phase` in the tlLogic's
    original order (cycling back to phase 0 if `green_phase` is last). Every junction
    inspected in this project so far (both the sandbox and the real Manhattan district)
    follows this convention -- SUMO's own default TLS generation always inserts a yellow
    phase directly after each green phase. Raises rather than guessing if a network
    doesn't follow it, since silently executing an unsafe abrupt phase change is worse
    than failing loudly.
    """
    next_index = (green_phase + 1) % len(phase_states)
    if not is_yellow_phase(phase_states[next_index]):
        raise ValueError(
            f"Phase {next_index} (immediately after green phase {green_phase}) is not "
            "yellow -- this network doesn't follow the green-then-yellow convention this "
            "function assumes."
        )
    return next_index
