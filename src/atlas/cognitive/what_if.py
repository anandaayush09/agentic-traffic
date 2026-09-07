"""The what-if tool: the Supervisor's VERIFY step (docs/design/04_agent_architecture.md
section 4.4: "every claim about consequences must cite a what-if run id -- the graph
enforces tool-grounding by construction"). Real simulation-backed verification for the
two playbook types this project can actually model with existing infrastructure;
everything else is honestly reported as unverifiable, not given a fabricated effect.

`switch_to_max_pressure`'s what-if directly reuses the same real-controller comparison
scripts/eval_smoke_suite.py already runs for its CI regression gate (max-pressure vs
fixed-time, real junctions, real network) -- answering the actual question this
playbook proposes: would falling back to max-pressure at these junctions measurably
help versus the classical fixed-time fallback underneath it. `hold_current_plan` is
trivially safe (no change proposed, nothing to verify). Every OTHER playbook type
(divert_arterial, meter_upstream, bus_bridge, weather_capacity_derate) would need a
real playbook-to-simulation effect compiler this project doesn't have yet -- `checked=
False` for those is the honest answer, not a guessed effect, and the Supervisor graph
(supervisor.py) treats "unchecked" as "always require operator approval", never
"assume it's fine".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from atlas.cognitive.playbooks import PlaybookProposal
from atlas.control.max_pressure import MaxPressureProgram
from atlas.services.decision import select_action_for_mode


@dataclass(frozen=True, slots=True)
class WhatIfResult:
    checked: bool
    """Whether this playbook type could actually be simulated -- False means the
    Supervisor has no grounded evidence either way, not that the playbook is bad."""
    improved: bool | None
    """None when `checked` is False. Otherwise: did the proposed playbook's real
    simulated mean pressure come out lower than (or equal to) the fixed-time
    baseline's."""
    baseline_mean_pressure: float | None
    proposed_mean_pressure: float | None


def run_what_if(
    proposal: PlaybookProposal,
    net_file: Path,
    route_file: Path,
    num_decisions: int = 40,
    seed: int = 42,
) -> WhatIfResult:
    if proposal.playbook_name == "hold_current_plan":
        return WhatIfResult(
            checked=True, improved=True, baseline_mean_pressure=None, proposed_mean_pressure=None
        )

    if proposal.playbook_name == "switch_to_max_pressure":
        junction_ids = proposal.params.get("junction_ids")
        if not isinstance(junction_ids, list) or not junction_ids:
            return WhatIfResult(
                checked=False,
                improved=None,
                baseline_mean_pressure=None,
                proposed_mean_pressure=None,
            )
        return _compare_max_pressure_vs_fixed_time(
            [str(j) for j in junction_ids], net_file, route_file, num_decisions, seed
        )

    return WhatIfResult(
        checked=False, improved=None, baseline_mean_pressure=None, proposed_mean_pressure=None
    )


def _compare_max_pressure_vs_fixed_time(
    junction_ids: list[str], net_file: Path, route_file: Path, num_decisions: int, seed: int
) -> WhatIfResult:
    from atlas.control.fixed_time import native_green_durations_from_net_file
    from atlas.control.shield import ShieldConfig
    from atlas.rl import MultiAgentSignalEnv, MultiAgentSignalEnvConfig
    from atlas.twin import TwinEnvConfig

    programs = {jid: MaxPressureProgram.from_net_file(jid, net_file) for jid in junction_ids}
    native_durations = {
        jid: native_green_durations_from_net_file(jid, net_file) for jid in junction_ids
    }
    max_lanes = max(len({m.from_lane for m in p.movements}) for p in programs.values())

    def make_env() -> MultiAgentSignalEnv:
        return MultiAgentSignalEnv(
            MultiAgentSignalEnvConfig(
                twin_config=TwinEnvConfig(name="what_if", net_file=net_file, route_file=route_file),
                tls_ids=junction_ids,
                max_lanes=max_lanes,
                max_green_phases=2,
                shield_config=ShieldConfig(min_green_s=10.0, max_green_s=60.0),
                decision_interval_s=5.0,
            )
        )

    def run(mode: str) -> float:
        env = make_env()
        env.reset(seed=seed)
        last_actions = dict.fromkeys(junction_ids, 0)
        total_pressure = 0.0
        try:
            for _ in range(num_decisions):
                conn = env.connection
                actions: dict[str, int] = {}
                for jid in junction_ids:
                    program = programs[jid]
                    queue_lengths = None
                    sim_time = None
                    if mode == "max_pressure":
                        lanes = {m.from_lane for m in program.movements} | {
                            m.to_lane for m in program.movements
                        }
                        queue_lengths = {
                            lane_id: int(conn.lane.getLastStepHaltingNumber(lane_id))
                            for lane_id in lanes
                        }
                    else:
                        sim_time = env.sim_time
                    actions[jid] = select_action_for_mode(
                        mode,  # type: ignore[arg-type]
                        program,
                        queue_lengths=queue_lengths,
                        native_durations=native_durations[jid],
                        sim_time=sim_time,
                        last_action=last_actions[jid],
                    )
                last_actions = actions
                _, _, _, _, infos = env.step(actions)
                total_pressure += sum(info["pressure"] for info in infos.values())
        finally:
            env.close()
        return total_pressure / num_decisions

    baseline = run("fixed_time")
    proposed = run("max_pressure")
    return WhatIfResult(
        checked=True,
        improved=proposed <= baseline,
        baseline_mean_pressure=baseline,
        proposed_mean_pressure=proposed,
    )
