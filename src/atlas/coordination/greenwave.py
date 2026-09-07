"""Green-wave bandwidth maximization and spillback detection -- the Corridor
Coordinator's algorithmic core (docs/design/04_agent_architecture.md section 4.2:
"Algorithmic, not learned: classic bandwidth-maximization (green wave) offsets from
measured platoon speeds + a spillback detector"; explicitly NOT learned, per that
section's own reasoning, to keep the system explainable and give the RL layer a
curriculum-stable feature).

Scope note, matching fixed_time.py's own precedent (`compute_webster_green_splits` is
"the formula itself... NOT yet wired to real per-movement flow estimates... future
work"): these are the formulas themselves, tested here against hand-computed values.
Wiring them to a live corridor (real inter-junction distances from the built network,
real platoon-speed measurements from the twin) is the Corridor Coordinator service's job,
not this module's.
"""

from __future__ import annotations

from collections.abc import Sequence


def compute_greenwave_offsets(
    segment_distances_m: Sequence[float], platoon_speed_mps: float
) -> tuple[float, ...]:
    """Cumulative green-start offsets (seconds), one per junction along the corridor,
    relative to the first junction (offset[0] == 0.0): how much LATER each junction's
    green should start so a platoon leaving junction 0 at the start of its green arrives
    at junction i exactly as junction i's own green begins.

    `segment_distances_m[k]` is the distance between junction k and junction k+1, so
    `len(segment_distances_m)` must be `num_junctions - 1`. Returns
    `len(segment_distances_m) + 1` offsets.
    """
    if platoon_speed_mps <= 0:
        raise ValueError(f"platoon_speed_mps must be positive, got {platoon_speed_mps}")
    if any(distance < 0 for distance in segment_distances_m):
        raise ValueError("segment distances must all be non-negative")

    offsets = [0.0]
    cumulative_s = 0.0
    for distance_m in segment_distances_m:
        cumulative_s += distance_m / platoon_speed_mps
        offsets.append(cumulative_s)
    return tuple(offsets)


def detect_spillback(exit_link_occupancy: float, threshold: float = 0.85) -> bool:
    """A junction is spilling back onto its upstream neighbor when its exit link's
    occupancy (fraction of length covered by vehicles, same metric as
    `atlas.twin.env.LaneState.occupancy`) exceeds `threshold` -- vehicles queued back
    past the junction itself, the corridor-level failure mode a single junction's own
    local view can't see (its own approach lanes look fine; the problem is downstream).
    """
    if not 0.0 <= exit_link_occupancy <= 1.0:
        raise ValueError(f"exit_link_occupancy must be in [0, 1], got {exit_link_occupancy}")
    return exit_link_occupancy >= threshold
