"""Gap-based actuated signal control -- docs/design/11_evaluation.md section 11.1's
second baseline: "SUMO-actuated -- gap-based green extension; representative of most
real deployments." Built and hand-verified here rather than delegated to SUMO's own
native `type="actuated"` tlLogic, for the same reason max-pressure and fixed-time are
hand-built (see their own module docstrings): the frozen architecture's control loop
already works at fixed `decision_interval_s` ticks with one pure "pick a phase"
function per mode (`atlas.services.decision.select_action_for_mode`), and this
controller needs to fit that exact interface, not a separately-configured native SUMO
program a Python-side agent has no visibility into.

The control law: extend the current green phase while (a) it hasn't reached
`max_green_s` yet AND (b) at least one vehicle is present on one of its own approach
lanes (a "gap" -- if no vehicle is there, there's nothing left to serve); once neither
condition holds, advance to the next green phase in the network's own defined cycle
order. `min_green_s` is enforced first and unconditionally, matching every real
actuated controller and this project's own `ShieldConfig.min_green_s` invariant.

Honest scope note: real actuated controllers re-evaluate every SUMO simulation step
(sub-second gap detection); this project's agents only get to make a decision every
`decision_interval_s` (5s in every config used so far), so "gap" here means "no
vehicle detected during the last decision interval," a coarser granularity than a
real actuated controller's continuous detection. This is a genuine, named
simplification forced by the frozen architecture's own decision cadence, not
something to silently claim is identical to a real deployment's actuated logic.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from atlas.control.max_pressure import MaxPressureProgram


@dataclass(frozen=True, slots=True)
class ActuatedConfig:
    min_green_s: float = 10.0
    max_green_s: float = 60.0


def select_actuated_phase(
    program: MaxPressureProgram,
    current_phase: int,
    elapsed_in_phase_s: float,
    queue_lengths: Mapping[str, int],
    config: ActuatedConfig | None = None,
) -> int:
    """`program` is reused as a general junction movement/phase structure here, not
    max-pressure-specific despite its name (the same movement/phase parsing every
    classical controller in this project needs). `queue_lengths` maps lane_id ->
    vehicle count present in the last decision interval, same shape max-pressure
    already consumes -- "present" here only needs > 0, not the exact count.

    Cycles through `program.green_phase_indices` in their own fixed order (never
    picks an out-of-turn phase, unlike max-pressure) -- gap-based extension decides
    only WHEN to move on, never WHICH phase comes next, matching real actuated
    controllers.
    """
    cfg = config or ActuatedConfig()
    if elapsed_in_phase_s < cfg.min_green_s:
        return current_phase

    vehicle_present = any(
        queue_lengths.get(m.from_lane, 0) > 0 for m in program.enabled_movements(current_phase)
    )
    if vehicle_present and elapsed_in_phase_s < cfg.max_green_s:
        return current_phase

    current_index = program.green_phase_indices.index(current_phase)
    next_index = (current_index + 1) % len(program.green_phase_indices)
    return program.green_phase_indices[next_index]
