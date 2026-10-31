#!/usr/bin/env python
"""Headless city benchmark: how much traffic can this machine simulate with ATLAS controlling it?

    python scripts/benchmark_city.py --city manhattan --period pm_peak --scale 0.25 --mode max_pressure \
        --sim-seconds 600 [--max-wall-s 900]

Prints one JSON line per stage and a final summary: setup time per stage, ATLAS decision-step cost,
achieved speed (simulated seconds per wall second, i.e. "x real time"), peak working-set RAM, and
the traffic outcome (vehicles in network, mean speed, teleports). A wall-clock budget stops the run
cleanly, so a too-slow configuration reports "slower than X real time" instead of hanging.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import libsumo  # noqa: E402,F401  (must precede sqlite3/pyarrow/pyproj imports on Windows)

from atlas.common.config import project_root  # noqa: E402


def rss_mb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1e6
    except ImportError:  # pragma: no cover
        return -1.0


def emit(stage: str, **fields: object) -> None:
    print(json.dumps({"stage": stage, "rss_mb": round(rss_mb()), **fields}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--city", default="manhattan")
    parser.add_argument("--period", default="pm_peak")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--mode", default="max_pressure")
    parser.add_argument("--sim-seconds", type=float, default=600)
    parser.add_argument("--teleport-s", type=float, default=300)
    parser.add_argument("--max-wall-s", type=float, default=900)
    parser.add_argument("--max-signals", type=int, default=0, help="0 = all controllable")
    args = parser.parse_args()

    from atlas.control.signal_index import NetworkSignalIndex
    from atlas.twin.controlled_twin import ControlledTwin
    from atlas.twin.incident_effects import RoadIndex

    city_dir = project_root() / "data" / "networks" / args.city
    net = city_dir / "net.net.xml"
    route = city_dir / "demand" / f"{args.period}.rou.xml"
    vtypes = [p for p in (city_dir / "vtypes.add.xml",) if p.is_file()]

    t = time.monotonic()
    index = NetworkSignalIndex.from_net_file(net)
    emit("signal_index", seconds=round(time.monotonic() - t, 1), signals=len(index))

    ids = index.controllable_ids
    if args.max_signals:
        ids = ids[: args.max_signals]
    t = time.monotonic()
    roads = RoadIndex.from_net_file(net)
    emit("road_index", seconds=round(time.monotonic() - t, 1))

    twin = ControlledTwin(
        net,
        route,
        vtypes,
        tls_ids=ids,
        default_mode=args.mode,
        road_index=roads,
        signal_index=index,
        time_to_teleport_s=args.teleport_s,
        scale=args.scale,
    )
    t = time.monotonic()
    twin.reset(seed=1)
    emit("sumo_load", seconds=round(time.monotonic() - t, 1), controlled=len(ids))

    conn = twin.connection
    peak = rss_mb()
    started, steps = time.monotonic(), 0
    step_seconds: list[float] = []
    stop_reason = "completed"
    while twin.sim_time < args.sim_seconds:
        s = time.monotonic()
        twin.step_decisions(1)
        step_seconds.append(time.monotonic() - s)
        steps += 1
        peak = max(peak, rss_mb())
        wall = time.monotonic() - started
        if steps % 12 == 0:
            emit(
                "progress",
                sim_s=twin.sim_time,
                wall_s=round(wall, 1),
                x_real_time=round(twin.sim_time / wall, 2),
                vehicles=int(conn.vehicle.getIDCount()),
            )
        if wall > args.max_wall_s:
            stop_reason = "wall_budget"
            break

    wall = time.monotonic() - started
    vehicle_ids = conn.vehicle.getIDList()
    speeds = [conn.vehicle.getSpeed(v) for v in vehicle_ids[:3000]]
    teleports = int(conn.simulation.getStartingTeleportNumber())
    step_seconds.sort()
    emit(
        "summary",
        stop_reason=stop_reason,
        mode=args.mode,
        scale=args.scale,
        controlled=len(ids),
        sim_seconds=twin.sim_time,
        wall_seconds=round(wall, 1),
        x_real_time=round(twin.sim_time / wall, 2) if wall else None,
        decision_steps=steps,
        median_step_s=round(step_seconds[len(step_seconds) // 2], 2) if step_seconds else None,
        peak_rss_mb=round(peak),
        vehicles_in_network=len(vehicle_ids),
        mean_speed_mps=round(sum(speeds) / len(speeds), 2) if speeds else None,
        teleports_last_step=teleports,
    )
    twin.close()


if __name__ == "__main__":
    main()
