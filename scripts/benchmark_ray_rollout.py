#!/usr/bin/env python
"""Measures achievable simulation throughput (steps/hour) parallelizing
MultiAgentSignalEnv across Ray Core rollout workers -- the Week 4 roadmap's "go/no-go
checkpoint against R2" (docs/design/12_risks.md's Colab/Kaggle throughput risk): this
number is what determines whether the Week 5-6 MAPPO training plan is realistic on
available compute.

Sweeps worker counts (1, 2, 4, ..., up to the local CPU count) rather than reporting a
single number, so the benchmark shows actual parallel scaling, not just one throughput
figure. Measured on whatever machine runs this script -- for this project, a local
Windows dev machine, not literally Kaggle; the roadmap's real training run happens on
Kaggle later, and this number is a same-order-of-magnitude planning proxy for that, not a
substitute for measuring on the actual training hardware.

Usage:
    python scripts/benchmark_ray_rollout.py [--network manhattan_midtown] [--steps 200]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import ray  # noqa: E402

from atlas.common.config import project_root  # noqa: E402
from atlas.common.logging import configure_logging, get_logger  # noqa: E402
from atlas.control.shield import ShieldConfig  # noqa: E402
from atlas.rl import MultiAgentSignalEnvConfig  # noqa: E402
from atlas.rl.rollout import RolloutWorker  # noqa: E402
from atlas.twin import TwinEnvConfig  # noqa: E402

configure_logging()
logger = get_logger(__name__)

# Every junction in both networks has exactly 2 green phases (verified in Week 4's
# multi-agent env work); max lane count differs, so it's supplied per network below.
NETWORKS: dict[str, dict[str, object]] = {
    "sandbox": {
        "tls_ids": ["A1", "B0", "B1", "B2"],
        "max_lanes": 4,
        "route_file": "routes.rou.xml",
    },
    "manhattan_midtown": {
        "tls_ids": [
            "42428283",
            "42428287",
            "42428297",
            "42430333",
            "42430338",
            "42430342",
            "42430344",
            "42430347",
            "42430350",
            "42430352",
            "42430356",
            "42432700",
            "42435663",
            "42435671",
            "42437363",
            "42439968",
            "42439972",
            "42439984",
            "42439990",
            "42439994",
            "5849918504",
            "GS_42439981",
        ],
        "max_lanes": 10,
        "route_file": "demand/pm_peak.rou.xml",
    },
}


def make_config(network: str) -> MultiAgentSignalEnvConfig:
    spec = NETWORKS[network]
    network_dir = project_root() / "data" / "networks" / network
    net_file = network_dir / "net.net.xml"
    route_file = network_dir / str(spec["route_file"])
    if not net_file.is_file() or not route_file.is_file():
        raise RuntimeError(
            f"{net_file} or {route_file} missing. Build/generate the '{network}' network "
            "and demand first."
        )
    # manhattan_midtown's demand is fleet-mixed (Week 2): vehicles reference vType ids
    # that only resolve via this additional file. The sandbox network's demand isn't
    # fleet-mixed, so this is empty there -- passing a nonexistent file would itself error.
    vtypes_file = network_dir / "vtypes.add.xml"
    additional_files = [vtypes_file] if vtypes_file.is_file() else []

    return MultiAgentSignalEnvConfig(
        # backend="traci": libsumo fails to import inside a Ray worker process on
        # Windows (a real native DLL collision, verified -- see
        # src/atlas/rl/rollout.py's module docstring for the full diagnosis). This
        # means the throughput measured here is a pessimistic lower bound, not
        # evidence of libsumo-in-Ray-workers' ceiling on Linux/Kaggle.
        twin_config=TwinEnvConfig(
            name="ray_benchmark",
            net_file=net_file,
            route_file=route_file,
            backend="traci",
            additional_files=additional_files,
        ),
        tls_ids=list(spec["tls_ids"]),  # type: ignore[arg-type]
        max_lanes=int(spec["max_lanes"]),  # type: ignore[arg-type]
        max_green_phases=2,
        shield_config=ShieldConfig(min_green_s=10.0, max_green_s=60.0),
    )


def run_at_worker_count(config: MultiAgentSignalEnvConfig, num_workers: int, steps: int) -> float:
    workers = [RolloutWorker.remote(config) for _ in range(num_workers)]  # type: ignore[attr-defined]
    try:
        results = ray.get(
            [w.collect_random_rollout.remote(steps, seed=i) for i, w in enumerate(workers)]
        )
    finally:
        for w in workers:
            ray.get(w.close.remote())

    total_steps = sum(r["n_steps"] for r in results)
    max_wall_clock_s = max(r["wall_clock_s"] for r in results)  # bottlenecked by the slowest
    return (total_steps / max_wall_clock_s) * 3600.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", choices=list(NETWORKS), default="manhattan_midtown")
    parser.add_argument("--steps", type=int, default=200, help="decisions per worker per point")
    args = parser.parse_args()

    config = make_config(args.network)
    num_agents = len(config.tls_ids)

    ray.init(ignore_reinit_error=True)
    try:
        available_cpus = int(ray.cluster_resources().get("CPU", 1))
        worker_counts = sorted({1, 2, 4, available_cpus})
        worker_counts = [w for w in worker_counts if w <= available_cpus]

        print(f"\n=== Ray rollout throughput: {args.network} ({num_agents} agents) ===")
        print(
            "NOTE: backend=traci (libsumo fails inside Ray workers on this Windows "
            "machine -- see src/atlas/rl/rollout.py docstring). These numbers are a "
            "pessimistic lower bound, not libsumo's ceiling; re-check on Kaggle/Linux."
        )
        print(f"{'workers':>8}{'steps/hour':>16}{'speedup_vs_1':>16}")
        baseline: float | None = None
        for n in worker_counts:
            steps_per_hour = run_at_worker_count(config, n, args.steps)
            if baseline is None:
                baseline = steps_per_hour
            speedup = steps_per_hour / baseline
            print(f"{n:>8}{steps_per_hour:>16,.0f}{speedup:>15.2f}x")
            logger.info(
                "benchmark.point", network=args.network, workers=n, steps_per_hour=steps_per_hour
            )
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
