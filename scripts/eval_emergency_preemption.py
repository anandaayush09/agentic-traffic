#!/usr/bin/env python
"""Measures the Preemption Overlay's effect on emergency-vehicle response time and
post-preemption network recovery -- the Week 7 roadmap's "measured emergency response
time improvement + recovery time" deliverable (docs/design/13_roadmap.md).

Runs the same corridor (the 5 manhattan_midtown junctions Week 5 trains on), same
demand, same seed, twice: once with every junction under plain max-pressure control
(baseline -- no preemption engaged), once with the Preemption Overlay actively locking
and forcing green along the emergency vehicle's route as it approaches each junction.
The only difference between the two runs is whether the overlay is engaged -- same
random seed, same demand file, same decision cadence.

The emergency route's edges (127693103#0/#3, 464516471#0/#3) and the green phase each
junction must be forced to for the emergency's approach (2, 0, 2, 0 respectively) were
verified directly against the built network's movement structure before writing this
script, the same way scripts/train_mappo_corridor.py verified its corridor's geometry.

Measures:
- response_time_s: sim-time from the emergency vehicle's departure to its arrival
  (disappearance from the simulation) -- lower is better.
- recovery_time_s: sim-time from the last lock's release until total corridor pressure
  first returns to within `recovery_tolerance` of its own pre-emergency baseline mean --
  a simple instantaneous threshold, not a smoothed/rolling estimate (documented scope,
  matching this project's "the metric itself first" pattern already used for Webster's
  formula and green-wave offsets).

Usage:
    python scripts/eval_emergency_preemption.py [--num-decisions 200]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atlas.common.config import project_root  # noqa: E402
from atlas.common.logging import configure_logging, get_logger  # noqa: E402
from atlas.control.max_pressure import MaxPressureProgram  # noqa: E402
from atlas.control.shield import ShieldConfig  # noqa: E402
from atlas.coordination.corridor_geometry import compute_route_etas_s  # noqa: E402
from atlas.coordination.lock_store import InMemoryLockStore  # noqa: E402
from atlas.coordination.preemption import PreemptionOverlay, RouteStop  # noqa: E402
from atlas.rl import MultiAgentSignalEnv, MultiAgentSignalEnvConfig  # noqa: E402
from atlas.services.decision import select_action_for_mode  # noqa: E402
from atlas.twin import TwinEnvConfig  # noqa: E402
from atlas.twin.emergency import EmergencyVehicleInjector, EmergencyVehicleSpec  # noqa: E402

configure_logging()
logger = get_logger(__name__)

NETWORK_DIR = project_root() / "data" / "networks" / "manhattan_midtown"
NET_FILE = NETWORK_DIR / "net.net.xml"
CORRIDOR_TLS_IDS = ["42430333", "42430338", "42430342", "42430344", "42430347"]
ROUTE_EDGES = ["127693103#0", "127693103#3", "464516471#0", "464516471#3"]
LOCKED_JUNCTIONS = CORRIDOR_TLS_IDS[1:]
FORCED_PHASES = {"42430338": 2, "42430342": 0, "42430344": 2, "42430347": 0}
DECISION_INTERVAL_S = 5.0
EMERGENCY_SPEED_MPS = 15.0
DEPART_S = 60.0
CLEARANCE_S = 15.0
MAX_LANES = 10
MAX_GREEN_PHASES = 2
RECOVERY_TOLERANCE = 0.10


@dataclass(frozen=True, slots=True)
class RunResult:
    response_time_s: float | None
    recovery_time_s: float | None
    baseline_mean_pressure: float


def make_env() -> MultiAgentSignalEnv:
    route_file = NETWORK_DIR / "demand" / "pm_peak.rou.xml"
    vtypes_file = NETWORK_DIR / "vtypes.add.xml"
    config = MultiAgentSignalEnvConfig(
        twin_config=TwinEnvConfig(
            name="eval_emergency",
            net_file=NET_FILE,
            route_file=route_file,
            additional_files=[vtypes_file] if vtypes_file.is_file() else [],
        ),
        tls_ids=CORRIDOR_TLS_IDS,
        max_lanes=MAX_LANES,
        max_green_phases=MAX_GREEN_PHASES,
        shield_config=ShieldConfig(min_green_s=10.0, max_green_s=60.0),
        decision_interval_s=DECISION_INTERVAL_S,
    )
    return MultiAgentSignalEnv(config)


def run(with_preemption: bool, num_decisions: int, seed: int = 42) -> RunResult:
    env = make_env()
    programs = {tid: MaxPressureProgram.from_net_file(tid, NET_FILE) for tid in CORRIDOR_TLS_IDS}
    env.reset(seed=seed)

    injector = EmergencyVehicleInjector(
        env,
        EmergencyVehicleSpec(
            vehicle_id="emg1",
            route_edges=ROUTE_EDGES,
            depart_s=DEPART_S,
            depart_speed=str(EMERGENCY_SPEED_MPS),
        ),
    )
    overlay: PreemptionOverlay | None = None
    route_stops: list[RouteStop] = []
    if with_preemption:
        etas_s = compute_route_etas_s(NET_FILE, ROUTE_EDGES, EMERGENCY_SPEED_MPS)
        route_stops = [
            RouteStop(tls_id=tid, phase_id=FORCED_PHASES[tid], eta_s=eta_s)
            for tid, eta_s in zip(LOCKED_JUNCTIONS, etas_s, strict=True)
        ]
        overlay = PreemptionOverlay(InMemoryLockStore(), clearance_s=CLEARANCE_S)

    last_actions = dict.fromkeys(CORRIDOR_TLS_IDS, 0)
    depart_sim_time: float | None = None
    run_triggered = False
    was_active = False
    response_time_s: float | None = None
    last_release_sim_time: float | None = None
    pressures_pre_emergency: list[float] = []
    pressures_after: list[tuple[float, float]] = []

    try:
        for _ in range(num_decisions):
            conn = env.connection
            actions: dict[str, int] = {}
            for tid in CORRIDOR_TLS_IDS:
                program = programs[tid]
                forced_phase = overlay.active_phase_for(tid) if overlay is not None else None
                if forced_phase is not None:
                    actions[tid] = program.green_phase_indices.index(forced_phase)
                    continue
                lanes = {m.from_lane for m in program.movements} | {
                    m.to_lane for m in program.movements
                }
                queue_lengths = {
                    lane_id: int(conn.lane.getLastStepHaltingNumber(lane_id)) for lane_id in lanes
                }
                actions[tid] = select_action_for_mode(
                    "max_pressure",
                    program,
                    queue_lengths=queue_lengths,
                    last_action=last_actions[tid],
                )
            last_actions = actions

            _, _, _, _, infos = env.step(actions)
            total_pressure = sum(info["pressure"] for info in infos.values())

            injector.maybe_inject()
            if injector.injected and depart_sim_time is None:
                depart_sim_time = env.sim_time

            if overlay is not None and injector.injected and depart_sim_time is not None:
                if not run_triggered:
                    overlay.trigger("emg1", route_stops)
                    run_triggered = True
                run = overlay.advance("emg1", now_s=env.sim_time - depart_sim_time)
                if run.state == "done" and last_release_sim_time is None:
                    last_release_sim_time = env.sim_time

            is_active = injector.is_active()
            if not injector.injected:
                pressures_pre_emergency.append(total_pressure)
            elif (
                was_active
                and not is_active
                and response_time_s is None
                and depart_sim_time is not None
            ):
                response_time_s = env.sim_time - depart_sim_time
            if injector.injected:
                pressures_after.append((env.sim_time, total_pressure))
            was_active = is_active
    finally:
        env.close()

    baseline_mean = (
        sum(pressures_pre_emergency) / len(pressures_pre_emergency)
        if pressures_pre_emergency
        else 0.0
    )
    recovery_time_s = None
    if last_release_sim_time is not None:
        threshold = baseline_mean + abs(baseline_mean) * RECOVERY_TOLERANCE + 1.0
        for sim_time, pressure in pressures_after:
            if sim_time >= last_release_sim_time and pressure <= threshold:
                recovery_time_s = sim_time - last_release_sim_time
                break

    return RunResult(
        response_time_s=response_time_s,
        recovery_time_s=recovery_time_s,
        baseline_mean_pressure=baseline_mean,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-decisions", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger.info("eval_emergency.baseline_run.starting")
    baseline = run(with_preemption=False, num_decisions=args.num_decisions, seed=args.seed)
    logger.info("eval_emergency.preemption_run.starting")
    preempted = run(with_preemption=True, num_decisions=args.num_decisions, seed=args.seed)

    print("\n=== Emergency preemption evaluation (5-junction corridor) ===")
    print(f"{'metric':<28}{'no preemption':>18}{'with preemption':>18}")
    print(
        f"{'response_time_s':<28}"
        f"{_fmt(baseline.response_time_s):>18}{_fmt(preempted.response_time_s):>18}"
    )
    print(
        f"{'recovery_time_s':<28}"
        f"{_fmt(baseline.recovery_time_s):>18}{_fmt(preempted.recovery_time_s):>18}"
    )

    if baseline.response_time_s is not None and preempted.response_time_s is not None:
        improvement_s = baseline.response_time_s - preempted.response_time_s
        print(f"\nresponse time improvement: {improvement_s:+.1f}s (positive = preemption helped)")
    else:
        print(
            "\nresponse time improvement: N/A (vehicle did not complete its route in "
            "both runs -- try a larger --num-decisions)"
        )


def _fmt(value: float | None) -> str:
    return f"{value:.1f}" if value is not None else "N/A"


if __name__ == "__main__":
    main()
