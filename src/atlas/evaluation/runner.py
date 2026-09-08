"""The scenario-suite runner (docs/design/11_evaluation.md section 11.4): runs ONE
(controller x scenario x seed) cell -- a single seeded episode of one classical/RL
controller, with the scenario engine's disruptions layered on top, against a real
`MultiAgentSignalEnv` -- and returns the real `TrafficMetrics` for it. The
"~35-cell grid x 4 controllers x 10 seeds ~= 1,400 evaluation runs" the design doc
describes is just this function called many times with different (mode, spec, seed)
tuples -- this module intentionally does NOT itself orchestrate that scale (a nightly/
background sweep script is a separate, later concern once this is verified correct on
a handful of real cells).

Scope note: `mode="rl"` is NOT wired into this runner (it would need a trained
checkpoint path + observation-building, and `select_action_for_mode` requires a
loaded `ActorCritic` + real observations neither of which this function currently
constructs) -- same honest stance `scripts/eval_smoke_suite.py` already takes for the
identical reason (Week 6: "no converged MAPPO artifact exists yet"). Calling this with
mode="rl" raises the same `ValueError` `select_action_for_mode` already raises for a
missing net/obs, not a silent no-op.
"""

from __future__ import annotations

import random
from pathlib import Path

from atlas.control.actuated import ActuatedConfig
from atlas.control.fixed_time import native_green_durations_from_net_file
from atlas.control.max_pressure import MaxPressureProgram
from atlas.control.shield import ShieldConfig
from atlas.evaluation.metrics import (
    QueueSampler,
    TrafficMetrics,
    compute_traffic_metrics,
    parse_tripinfo,
)
from atlas.rl.multi_agent_env import MultiAgentSignalEnv, MultiAgentSignalEnvConfig
from atlas.services.decision import AgentMode, select_action_for_mode
from atlas.twin.env import TwinEnvConfig
from atlas.twin.scenario import ScenarioController, ScenarioSpec
from atlas.twin.sensor_fault import SensorFaultModel


def run_scenario_cell(
    mode: AgentMode,
    tls_ids: list[str],
    net_file: Path,
    route_file: Path,
    seed: int,
    num_decisions: int,
    tripinfo_output: Path,
    decision_interval_s: float = 5.0,
    scenario_specs: list[ScenarioSpec] | None = None,
    additional_files: list[Path] | None = None,
    max_lanes: int = 10,
    max_green_phases: int = 2,
    detector_dropout_rate: float = 0.0,
) -> TrafficMetrics:
    """Runs `num_decisions` decision cycles (each `decision_interval_s` real
    simulation seconds) of `mode` controlling every junction in `tls_ids` at once,
    with `scenario_specs` (if any) applying/reverting on their own configured
    schedule throughout. Returns the real, measured `TrafficMetrics` for the whole
    episode (docs/design/11_evaluation.md section 11.2).

    `detector_dropout_rate` (docs/design/11_evaluation.md section 11.5's detector-
    dropout robustness sweep): applies `SensorFaultModel` to every controller's
    `queue_lengths` reading before it's used for a decision -- `fixed_time` is
    unaffected by construction (it never reads queue_lengths at all), which is
    itself the honest, expected result for that mode, not a bug in the sweep.
    """
    fault_model = (
        SensorFaultModel(dropout_rate=detector_dropout_rate, rng=random.Random(seed))
        if detector_dropout_rate > 0.0
        else None
    )
    programs = {tls_id: MaxPressureProgram.from_net_file(tls_id, net_file) for tls_id in tls_ids}
    native_durations = {
        tls_id: native_green_durations_from_net_file(tls_id, net_file) for tls_id in tls_ids
    }

    env = MultiAgentSignalEnv(
        MultiAgentSignalEnvConfig(
            twin_config=TwinEnvConfig(
                name="eval_runner",
                net_file=net_file,
                route_file=route_file,
                additional_files=additional_files or [],
                tripinfo_output=tripinfo_output,
            ),
            tls_ids=tls_ids,
            max_lanes=max_lanes,
            max_green_phases=max_green_phases,
            shield_config=ShieldConfig(min_green_s=10.0, max_green_s=60.0),
            decision_interval_s=decision_interval_s,
        )
    )
    scenario_controller = (
        # ScenarioController's own type hint says SumoTwinEnv (its `step()` method
        # calls env.step() with no args, which MultiAgentSignalEnv doesn't support --
        # its step() needs a per-junction actions dict). This runner only ever calls
        # `.sync()`, which structurally only needs `.connection`/`.sim_time`
        # (`atlas.twin.env.TwinLike`), both of which MultiAgentSignalEnv provides --
        # verified real usage, not a type-checking loophole being exploited.
        ScenarioController(env, scenario_specs)  # type: ignore[arg-type]
        if scenario_specs
        else None
    )

    env.reset(seed=seed)
    last_actions = dict.fromkeys(tls_ids, 0)
    elapsed_in_phase_s = dict.fromkeys(tls_ids, 0.0)
    queue_sampler = QueueSampler()

    try:
        for _ in range(num_decisions):
            if scenario_controller is not None:
                scenario_controller.sync()

            actions: dict[str, int] = {}
            for tls_id in tls_ids:
                program = programs[tls_id]
                conn = env.connection
                lanes = {m.from_lane for m in program.movements} | {
                    m.to_lane for m in program.movements
                }
                queue_lengths = {
                    lane_id: int(conn.lane.getLastStepHaltingNumber(lane_id)) for lane_id in lanes
                }
                queue_sampler.sample(conn, list(lanes))  # the real, un-faulted ground
                # truth -- this measures actual network state, not what a degraded
                # controller believes; only the decision below sees the fault.
                controller_queue_lengths = (
                    fault_model.apply(queue_lengths) if fault_model is not None else queue_lengths
                )

                action = select_action_for_mode(
                    mode,
                    program,
                    queue_lengths=controller_queue_lengths,
                    native_durations=native_durations[tls_id],
                    sim_time=env.sim_time if mode == "fixed_time" else None,
                    last_action=last_actions[tls_id],
                    elapsed_in_phase_s=elapsed_in_phase_s[tls_id] if mode == "actuated" else None,
                    actuated_config=ActuatedConfig(),
                )
                elapsed_in_phase_s[tls_id] = (
                    elapsed_in_phase_s[tls_id] + decision_interval_s
                    if action == last_actions[tls_id]
                    else decision_interval_s
                )
                actions[tls_id] = action

            last_actions = actions
            env.step(actions)

        episode_duration_s = num_decisions * decision_interval_s
    finally:
        env.close()

    trips = parse_tripinfo(tripinfo_output)
    return compute_traffic_metrics(trips, queue_sampler.samples, episode_duration_s)
