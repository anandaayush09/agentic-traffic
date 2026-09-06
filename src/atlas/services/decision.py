"""The pure decision logic behind IntersectionAgentService's `mode` switch (docs/design/
04_agent_architecture.md section 4.1: "APIs: /health, /mode (rl | max_pressure | fixed |
off)") -- extracted from the live-service loop specifically so it's testable without a
running SUMO twin: `MaxPressureProgram.from_net_file` only parses the built network XML
(no libsumo needed), so every mode's action-selection can be verified against real
junction data in fast unit tests.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import torch

from atlas.control.fixed_time import select_fixed_time_phase
from atlas.control.max_pressure import MaxPressureProgram, select_max_pressure_phase
from atlas.marl.network import ActorCritic

AgentMode = Literal["rl", "max_pressure", "fixed_time", "off"]

_MODE_TO_ACTOR: dict[AgentMode, str] = {
    "rl": "rl_agent",
    "max_pressure": "max_pressure",
    "fixed_time": "fixed_time",
    "off": "fixed_time",  # "off" freezes the last commanded phase; audit as fixed_time
}


def actor_name_for_mode(mode: AgentMode) -> str:
    """Maps an agent mode to `control.commands.v1`'s `actor` enum value."""
    return _MODE_TO_ACTOR[mode]


def select_action_for_mode(
    mode: AgentMode,
    program: MaxPressureProgram,
    *,
    queue_lengths: Mapping[str, int] | None = None,
    native_durations: tuple[tuple[int, ...], tuple[float, ...]] | None = None,
    sim_time: float | None = None,
    net: ActorCritic | None = None,
    obs: torch.Tensor | None = None,
    last_action: int = 0,
) -> int:
    """Returns an action index into `program.green_phase_indices` -- the same index
    `MultiAgentSignalEnv.step()` expects -- for whichever mode is active. Each mode
    needs a different subset of the optional keyword arguments; the caller (the live
    service loop) is responsible for gathering the right ones from the running twin.
    """
    if mode == "off":
        return last_action

    if mode == "max_pressure":
        if queue_lengths is None:
            raise ValueError("max_pressure mode requires queue_lengths")
        desired_phase = select_max_pressure_phase(program, queue_lengths)
        return program.green_phase_indices.index(desired_phase)

    if mode == "fixed_time":
        if native_durations is None or sim_time is None:
            raise ValueError("fixed_time mode requires native_durations and sim_time")
        green_phase_indices, durations_s = native_durations
        desired_phase = select_fixed_time_phase(green_phase_indices, durations_s, sim_time)
        return program.green_phase_indices.index(desired_phase)

    if mode == "rl":
        if net is None or obs is None:
            raise ValueError("rl mode requires net and obs")
        with torch.no_grad():
            action, _, _ = net.act(obs.unsqueeze(0), deterministic=True)
        return int(action.item())

    raise ValueError(f"Unknown mode: {mode!r}")
