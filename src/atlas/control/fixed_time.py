"""Fixed-time signal control: Webster's method (1958) for computing an optimal cycle
length and green splits from per-phase critical flow ratios, plus a deterministic cycling
mechanism that executes whatever timing plan it's given. Two separable concerns on
purpose, matching real traffic-engineering practice: Webster's method produces a *timing
plan*; a signal controller just runs one. Serves as the weakest classical baseline (Part
11 evaluation) and the bottom rung of the safety shield's fallback ladder (RL ->
max-pressure -> fixed-time, Part 3.10) -- it needs no real-time data at all, so it is the
one thing that always works.

Scope note: `compute_webster_green_splits` is the formula itself, tested here against
hand-computed values -- it is NOT yet wired to real per-movement flow estimates from this
project's generated demand (docs/design/08_data_strategy.md's "calibrated demand" is
later evaluation-quality work). Feeding it real flow ratios is future work, not silently
assumed done.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

from atlas.control.phases import green_phase_indices as _green_phase_indices


def native_green_durations_from_net_file(
    tls_id: str, net_file: Path
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Reads a junction's own tlLogic green-phase durations directly from the built
    network -- the "as designed" fixed-time plan, as an alternative to Webster's method
    when no calibrated flow data is available yet. Returns (green_phase_indices,
    durations_s), same order, ready to pass to `select_fixed_time_phase`.
    """
    root = ElementTree.parse(net_file).getroot()
    logic = root.find(f".//tlLogic[@id='{tls_id}']")
    if logic is None:
        raise ValueError(f"No tlLogic found for traffic light '{tls_id}' in {net_file}")

    phases = logic.findall("phase")
    phase_states = [p.get("state", "") for p in phases]
    green = _green_phase_indices(phase_states)
    if not green:
        raise ValueError(f"No green phases found for traffic light '{tls_id}'")

    durations = tuple(float(phases[i].get("duration", "0")) for i in green)
    return green, durations


@dataclass(frozen=True, slots=True)
class WebsterResult:
    cycle_s: float
    green_durations_s: tuple[float, ...]
    """One duration per input flow ratio, same order."""


def compute_webster_green_splits(
    flow_ratios: Sequence[float],
    lost_time_per_phase_s: float = 4.0,
    min_green_s: float = 10.0,
) -> WebsterResult:
    """Webster's optimal cycle length and green splits.

    C0 = (1.5*L + 5) / (1 - Y), where L is total lost time across all phases and Y is the
    sum of critical flow ratios y_i = q_i/s_i (demand flow / saturation flow) for each
    phase's critical movement. Effective green is then split proportionally to each
    phase's y_i. `min_green_s` is a floor applied after the proportional split (a
    genuinely low-flow phase can otherwise be computed below any safe minimum).
    """
    if not flow_ratios:
        raise ValueError("flow_ratios must be non-empty")
    if any(y <= 0 for y in flow_ratios):
        raise ValueError("flow_ratios must all be positive")

    total_y = sum(flow_ratios)
    if total_y >= 1.0:
        raise ValueError(
            f"sum of flow ratios ({total_y}) must be < 1.0 (>=1.0 is an oversaturated "
            "intersection Webster's method does not apply to)"
        )

    num_phases = len(flow_ratios)
    total_lost_time_s = lost_time_per_phase_s * num_phases
    cycle_s = (1.5 * total_lost_time_s + 5.0) / (1.0 - total_y)
    effective_green_total_s = cycle_s - total_lost_time_s

    green_durations_s = tuple(
        max(min_green_s, (y / total_y) * effective_green_total_s) for y in flow_ratios
    )
    return WebsterResult(cycle_s=cycle_s, green_durations_s=green_durations_s)


def select_fixed_time_phase(
    green_phase_indices: Sequence[int],
    green_durations_s: Sequence[float],
    cycle_elapsed_s: float,
) -> int:
    """Which green phase should be active `cycle_elapsed_s` seconds into a repeating
    cycle, given a fixed per-phase duration plan (from `compute_webster_green_splits` or
    any other source, e.g. a network's own tlLogic durations). Purely a function of
    elapsed time -- no queue or other real-time data needed, which is exactly why this is
    the shield's last-resort fallback.
    """
    if len(green_phase_indices) != len(green_durations_s):
        raise ValueError("green_phase_indices and green_durations_s must be the same length")
    total_cycle_s = sum(green_durations_s)
    if total_cycle_s <= 0:
        raise ValueError("total cycle duration must be positive")

    t = cycle_elapsed_s % total_cycle_s
    cumulative_s = 0.0
    for phase, duration_s in zip(green_phase_indices, green_durations_s, strict=True):
        cumulative_s += duration_s
        if t < cumulative_s:
            return phase
    return green_phase_indices[-1]  # floating-point edge case at the exact cycle boundary
