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
