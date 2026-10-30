"""The what-if tool: the Supervisor's VERIFY step (docs/design/04_agent_architecture.md
section 4.4: "every claim about consequences must cite a what-if run id -- the graph
enforces tool-grounding by construction"). Real simulation-backed verification for the
two playbook types this project can actually model with existing infrastructure;
everything else is honestly reported as unverifiable, not given a fabricated effect.

`divert_arterial` and `meter_upstream` are verified by simulating the INCIDENT itself on
the real network with and without the playbook's response (`atlas.twin.controlled_twin`
+ `atlas.twin.playbook_effects`) and comparing mean halted vehicles network-wide -- so
they need the incident, not just the proposal. Without one (or when the playbook can't be
compiled onto this network) the answer stays honestly `checked=False`.

`switch_to_max_pressure`'s what-if directly reuses the same real-controller comparison
scripts/eval_smoke_suite.py already runs for its CI regression gate (max-pressure vs
fixed-time, real junctions, real network) -- answering the actual question this
playbook proposes: would falling back to max-pressure at these junctions measurably
help versus the classical fixed-time fallback underneath it. `hold_current_plan` is
trivially safe (no change proposed, nothing to verify). `bus_bridge` (no transit in the
network) and `weather_capacity_derate` (derating *assumed* capacity has no physical
lever) have no defensible simulation -- `checked=False` for those is the honest answer,
not a guessed effect, and the Supervisor graph (supervisor.py) treats "unchecked" as
"always require operator approval", never "assume it's fine".
"""

from __future__ import annotations

from collections.abc import Mapping
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
    metric: str = "mean_pressure"
    """What the two numbers above measure (lower is better): "mean_pressure" for the
    max-pressure comparison, "mean_halted_vehicles" for incident-response comparisons."""


# Simulated-time layout of an incident what-if: warm the network up, hit it with the
# incident (plus the response, in the proposed run), then measure. Decisions are 5 s each.
_WARMUP_DECISIONS = 60
_MEASURE_DECISIONS = 72
_DISRUPTION_DURATION_S = 900


def run_what_if(
    proposal: PlaybookProposal,
    net_file: Path,
    route_file: Path,
    num_decisions: int = 40,
    seed: int = 42,
    additional_files: list[Path] | None = None,
    incident: Mapping[str, object] | None = None,
) -> WhatIfResult:
    """`incident`: the incident being responded to -- required to verify
    `divert_arterial`/`meter_upstream`, which are judged by how they change the
    incident's impact. `additional_files`: SUMO additional files (e.g. a network's `vtypes.add.xml`)
    needed alongside `route_file`. Real gap this parameter closes: manhattan_midtown's
    demand routes reference vehicle types defined in a separate fleet-mix additional
    file (Week 2 part 4), not inline -- omitting it isn't a problem on the sandbox
    network (no separate vtypes file there), which is why this went unnoticed until
    exercised against manhattan_midtown for real via the Week 10 dashboard's demo
    incident flow (`libsumo.libsumo.TraCIException: The vehicle type
    'passenger_vtype'... is not known`).
    """
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
            [str(j) for j in junction_ids],
            net_file,
            route_file,
            num_decisions,
            seed,
            additional_files or [],
        )

    if proposal.playbook_name in ("divert_arterial", "meter_upstream") and incident is not None:
        return _compare_incident_response(
            proposal, incident, net_file, route_file, seed, additional_files or []
        )

    return WhatIfResult(
        checked=False, improved=None, baseline_mean_pressure=None, proposed_mean_pressure=None
    )


def _compare_incident_response(
    proposal: PlaybookProposal,
    incident: Mapping[str, object],
    net_file: Path,
    route_file: Path,
    seed: int,
    additional_files: list[Path],
) -> WhatIfResult:
    from atlas.twin.controlled_twin import CORRIDOR_TLS_IDS, ControlledTwin, edge_halted_total
    from atlas.twin.incident_effects import (
        ROAD_EFFECT_CLASSES,
        RoadIndex,
        link_ids_of,
        plan_disruption_specs,
    )
    from atlas.twin.playbook_effects import compile_playbook

    unverifiable = WhatIfResult(
        checked=False, improved=None, baseline_mean_pressure=None, proposed_mean_pressure=None
    )
    incident_class = str(incident.get("class", ""))
    if incident_class not in ROAD_EFFECT_CLASSES:
        return unverifiable  # nothing physical to respond to in the simulation

    index = RoadIndex.from_net_file(net_file)
    incident_edges = index.resolve(link_ids_of(incident.get("location")))
    effect = compile_playbook(proposal.playbook_name, proposal.params, index, CORRIDOR_TLS_IDS)
    if not incident_edges or effect is None:
        return unverifiable

    incident_id = str(incident.get("incident_id", "what_if"))
    all_edges = list(index.lane_counts)

    def run(with_response: bool) -> float:
        twin = ControlledTwin(
            net_file, route_file, additional_files, road_index=index, backend="libsumo"
        )
        twin.reset(seed=seed)
        try:
            twin.step_decisions(_WARMUP_DECISIONS)
            twin.add_disruptions(
                plan_disruption_specs(
                    incident_id,
                    incident_class,
                    {e: index.lane_counts[e] for e in incident_edges},
                    int(twin.sim_time),
                    _DISRUPTION_DURATION_S,
                )
            )
            if with_response:
                twin.apply_effect("what_if_response", effect)
            conn = twin.connection
            total = 0
            for _ in range(_MEASURE_DECISIONS):
                twin.step_decisions(1)
                total += edge_halted_total(conn, all_edges)
            return total / _MEASURE_DECISIONS
        finally:
            twin.close()

    baseline = run(with_response=False)
    proposed = run(with_response=True)
    return WhatIfResult(
        checked=True,
        improved=proposed < baseline,
        baseline_mean_pressure=baseline,
        proposed_mean_pressure=proposed,
        metric="mean_halted_vehicles",
    )


def _compare_max_pressure_vs_fixed_time(
    junction_ids: list[str],
    net_file: Path,
    route_file: Path,
    num_decisions: int,
    seed: int,
    additional_files: list[Path],
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
                twin_config=TwinEnvConfig(
                    name="what_if",
                    net_file=net_file,
                    route_file=route_file,
                    additional_files=additional_files,
                ),
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
