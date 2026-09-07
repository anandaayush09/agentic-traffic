#!/usr/bin/env python
"""Nightly CI regression gate -- the Week 6 roadmap's "eval.yml nightly mini-suite
(2-junction smoke + district vs fixed-time) as CI regression gate" (docs/design/
13_roadmap.md, Week 6).

Compares max-pressure against fixed-time on two scenarios, using the already-built
classical controllers rather than a trained RL checkpoint: no converged MAPPO artifact
exists yet -- Week 5/6's corridor and district training runs are documented as short
pipeline-verification runs, not the real overnight/weekend Kaggle convergence run
(docs/design/06_ai_pipeline.md section 6.1's own throughput math). Committing an
under-trained checkpoint as "the model this gate checks" would be worse than not gating
on RL at all -- a flaky, meaningless comparison dressed up as a regression gate. This
gate instead protects the layer that DOES have a hard guarantee: max-pressure is a
deterministic control law with a proven stability property (Varaiya 2013) and should
systematically produce lower network pressure than a naive fixed-time cycle under real
load. A regression here means something broke in max_pressure.py, shield.py, or
MultiAgentSignalEnv itself -- exactly the class of bug this gate exists to catch.
Swapping in a real trained RL checkpoint, once one exists from an actual Kaggle
convergence run, is a follow-up to this script, not a redesign of it.

Two scenarios:
1. "2-junction smoke" -- the sandbox network (2 of its 4 junctions), built fresh on every
   run (fast, no OSM/netconvert dependency).
2. "district vs fixed-time" -- the full manhattan_midtown district (22 junctions), built
   from the committed OSM extract exactly as the test suite already does.

Fails (non-zero exit) if max-pressure's mean network pressure is not lower than or equal
to fixed-time's, in either scenario -- a legitimate hard assertion given max-pressure's
theoretical guarantee, not a flaky statistical threshold.

Usage:
    python scripts/eval_smoke_suite.py [--num-decisions 100]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from atlas.common.config import project_root  # noqa: E402
from atlas.common.logging import configure_logging, get_logger  # noqa: E402
from atlas.control.fixed_time import native_green_durations_from_net_file  # noqa: E402
from atlas.control.max_pressure import MaxPressureProgram  # noqa: E402
from atlas.control.shield import ShieldConfig  # noqa: E402
from atlas.rl import MultiAgentSignalEnv, MultiAgentSignalEnvConfig  # noqa: E402
from atlas.services.decision import select_action_for_mode  # noqa: E402
from atlas.twin import TwinEnvConfig  # noqa: E402

configure_logging()
logger = get_logger(__name__)

DECISION_INTERVAL_S = 5.0


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    name: str
    max_pressure_mean: float
    fixed_time_mean: float

    @property
    def passed(self) -> bool:
        return self.max_pressure_mean <= self.fixed_time_mean


def _run_controller(
    env: MultiAgentSignalEnv,
    programs: dict[str, MaxPressureProgram],
    native_durations: dict[str, tuple[tuple[int, ...], tuple[float, ...]]],
    mode: str,
    num_decisions: int,
    seed: int,
) -> float:
    env.reset(seed=seed)
    total_pressure_sum = 0.0
    last_actions = dict.fromkeys(env.agents, 0)

    for _ in range(num_decisions):
        actions: dict[str, int] = {}
        for tls_id in env.agents:
            queue_lengths = None
            sim_time = None
            if mode == "max_pressure":
                conn = env.connection
                program = programs[tls_id]
                lanes = {m.from_lane for m in program.movements} | {
                    m.to_lane for m in program.movements
                }
                queue_lengths = {
                    lane_id: int(conn.lane.getLastStepHaltingNumber(lane_id)) for lane_id in lanes
                }
            elif mode == "fixed_time":
                sim_time = env.sim_time

            actions[tls_id] = select_action_for_mode(
                mode,  # type: ignore[arg-type]
                programs[tls_id],
                queue_lengths=queue_lengths,
                native_durations=native_durations[tls_id],
                sim_time=sim_time,
                last_action=last_actions[tls_id],
            )
        last_actions = actions

        _, _, _, _, infos = env.step(actions)
        total_pressure_sum += sum(info["pressure"] for info in infos.values())

    return total_pressure_sum / num_decisions


def run_scenario(
    name: str,
    net_file: Path,
    route_file: Path,
    tls_ids: list[str],
    max_lanes: int,
    additional_files: list[Path],
    num_decisions: int,
    seed: int = 42,
) -> ScenarioResult:
    programs = {tid: MaxPressureProgram.from_net_file(tid, net_file) for tid in tls_ids}
    native_durations = {tid: native_green_durations_from_net_file(tid, net_file) for tid in tls_ids}

    def make_env() -> MultiAgentSignalEnv:
        return MultiAgentSignalEnv(
            MultiAgentSignalEnvConfig(
                twin_config=TwinEnvConfig(
                    name=f"eval_smoke_{name}",
                    net_file=net_file,
                    route_file=route_file,
                    additional_files=additional_files,
                ),
                tls_ids=tls_ids,
                max_lanes=max_lanes,
                max_green_phases=2,
                shield_config=ShieldConfig(min_green_s=10.0, max_green_s=60.0),
                decision_interval_s=DECISION_INTERVAL_S,
            )
        )

    max_pressure_env = make_env()
    try:
        max_pressure_mean = _run_controller(
            max_pressure_env, programs, native_durations, "max_pressure", num_decisions, seed
        )
    finally:
        max_pressure_env.close()

    fixed_time_env = make_env()
    try:
        fixed_time_mean = _run_controller(
            fixed_time_env, programs, native_durations, "fixed_time", num_decisions, seed
        )
    finally:
        fixed_time_env.close()

    return ScenarioResult(name, max_pressure_mean, fixed_time_mean)


def run_sandbox_smoke(num_decisions: int) -> ScenarioResult:
    import generate_sandbox_network

    generate_sandbox_network.generate()
    network_dir = project_root() / "data" / "networks" / "sandbox"
    return run_scenario(
        "2_junction_smoke",
        net_file=network_dir / "net.net.xml",
        route_file=network_dir / "routes.rou.xml",
        tls_ids=["B0", "B1"],
        max_lanes=4,
        additional_files=[],
        num_decisions=num_decisions,
    )


def run_district_vs_fixed_time(num_decisions: int) -> ScenarioResult:
    import build_district_network
    import generate_demand

    build_district_network.build_network()
    generated = generate_demand.generate_all()
    network_dir = project_root() / "data" / "networks" / "manhattan_midtown"
    vtypes_file = network_dir / "vtypes.add.xml"

    import train_mappo_district

    return run_scenario(
        "district_vs_fixed_time",
        net_file=network_dir / "net.net.xml",
        route_file=generated["pm_peak"],
        tls_ids=train_mappo_district.DISTRICT_TLS_IDS,
        max_lanes=10,
        additional_files=[vtypes_file] if vtypes_file.is_file() else [],
        num_decisions=num_decisions,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-decisions", type=int, default=100)
    args = parser.parse_args()

    results = [
        run_sandbox_smoke(args.num_decisions),
        run_district_vs_fixed_time(args.num_decisions),
    ]

    print(f"\n{'scenario':<28}{'max_pressure':>16}{'fixed_time':>16}{'result':>10}")
    all_passed = True
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        all_passed &= result.passed
        print(
            f"{result.name:<28}{result.max_pressure_mean:>16.3f}"
            f"{result.fixed_time_mean:>16.3f}{status:>10}"
        )
        logger.info(
            "eval_smoke.scenario",
            name=result.name,
            max_pressure_mean=result.max_pressure_mean,
            fixed_time_mean=result.fixed_time_mean,
            passed=result.passed,
        )

    if not all_passed:
        print("\nREGRESSION: max-pressure did not beat fixed-time in at least one scenario.")
        sys.exit(1)
    print("\nAll scenarios passed.")


if __name__ == "__main__":
    main()
