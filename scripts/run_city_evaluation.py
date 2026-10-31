#!/usr/bin/env python
"""Controller comparison on a whole city (default: Manhattan, all controllable signals).

    python scripts/run_city_evaluation.py --city manhattan --scale 0.25 --seeds 3 --decisions 180

Runs each classical controller (fixed-time, actuated, max-pressure) over the same seeds on the
same network and demand, then writes a report with the same statistics as the Midtown pilot
(Welch's t-test, Holm-Bonferroni across the comparisons, Cohen's d). It is RESUMABLE: every
finished cell is appended to `data/networks/<city>/runs/evaluation_cells.jsonl`, and a re-run skips
cells already there, so an interrupted run loses at most the cell in progress.

What the numbers mean, stated plainly (they are repeated in the report):
- the demand is synthetic (randomTrips) and scaled by `--scale`;
- an episode starts from an EMPTY network, so a short episode measures how each controller handles
  the network filling up, not steady state;
- waiting / travel time are averaged over every trip that DEPARTED, including vehicles still en
  route when the episode ends (their time so far counts); throughput counts only finished trips;
- a handful of seeds gives wide intervals; the report states n.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import libsumo  # noqa: E402,F401  (must precede sqlite3/pyarrow/pyproj imports on Windows)

from atlas.common.config import project_root  # noqa: E402
from atlas.common.logging import configure_logging, get_logger  # noqa: E402
from atlas.control.signal_index import NetworkSignalIndex  # noqa: E402
from atlas.evaluation.runner import run_scenario_cell  # noqa: E402
from atlas.evaluation.statistics import compare_two_controllers, holm_bonferroni  # noqa: E402

configure_logging()
logger = get_logger(__name__)

CONTROLLERS = ["fixed_time", "actuated", "max_pressure"]
BASELINE = "fixed_time"
METRICS = (
    "mean_waiting_time_s",
    "mean_travel_time_s",
    "throughput_veh_per_hour",
    "mean_queue_length",
)


def load_cells(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def values(cells: list[dict[str, object]], controller: str, metric: str) -> list[float]:
    chosen = sorted(
        (c for c in cells if c["controller"] == controller),
        key=lambda c: int(c["seed"]),  # type: ignore[call-overload]
    )
    return [float(c[metric]) for c in chosen]  # type: ignore[arg-type]


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def build_report(cells: list[dict[str, object]], settings: dict[str, object]) -> str:
    controllers = [c for c in CONTROLLERS if any(x["controller"] == c for x in cells)]
    seeds_each = {c: len(values(cells, c, "mean_waiting_time_s")) for c in controllers}
    lines = [
        f"# {str(settings['city']).title()} controller comparison",
        "",
        f"Network: `{settings['city']}`, {settings['signals']} signals controlled by each controller",
        f"Demand: synthetic `pm_peak`, scaled x{settings['scale']}",
        f"Episode: {settings['decisions']} decisions = {int(settings['decisions']) * 5} simulated seconds, "  # type: ignore[call-overload]
        "starting from an empty network",
        f"Seeds per controller: {seeds_each}",
        f"Teleport after: {settings['teleport_s']} s",
        "",
        "## How to read this",
        "",
        "- Demand is synthetic, not calibrated to real counts.",
        "- The network starts empty, so a short episode measures how each controller copes while the "
        "network fills, not steady state.",
        "- Waiting and travel time are averaged over every trip that departed, including vehicles "
        "still en route at the end (their time so far counts). Throughput counts finished trips "
        "only; queue length is sampled every 30 s on every approach lane.",
        "- **Seeds only change SUMO's random choices (driver behaviour, insertion order); every run "
        "replays the same generated trips.** Variation between seeds is therefore far smaller than "
        "real day-to-day variation, so the confidence intervals, p-values and Cohen's d below are "
        'over-confident. Read them as "consistent on this demand", not as proof it generalises to '
        "other demand patterns. Independent demand draws would be needed for that.",
        "- p-values are Holm-Bonferroni corrected across the comparisons below.",
        "",
        "## Means per controller",
        "",
        "| Controller | n | Waiting (s) | Travel (s) | Throughput (veh/h) | Mean queue (veh/lane) |",
        "|---|---|---|---|---|---|",
    ]
    for c in controllers:
        m = {k: mean(values(cells, c, k)) for k in METRICS}
        lines.append(
            f"| {c} | {seeds_each[c]} | {m['mean_waiting_time_s']:.1f} | {m['mean_travel_time_s']:.1f} "
            f"| {m['throughput_veh_per_hour']:.0f} | {m['mean_queue_length']:.3f} |"
        )

    rows = []
    for metric in ("mean_waiting_time_s", "throughput_veh_per_hour"):
        base = values(cells, BASELINE, metric)
        for c in controllers:
            if c == BASELINE or len(base) < 2 or seeds_each[c] < 2:
                continue
            rows.append((metric, c, compare_two_controllers(base, values(cells, c, metric))))
    if rows:
        significant = holm_bonferroni([r[2].p_value for r in rows], alpha=0.05)
        lines += [
            "",
            f"## Comparison against {BASELINE}",
            "",
            "| Metric | Controller | Mean | Baseline mean | 95% CI | p-value | Significant (Holm)? | Cohen's d |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for (metric, c, cmp), sig in zip(rows, significant, strict=True):
            ci = cmp.treatment_ci
            lines.append(
                f"| {metric} | {c} | {ci.mean:.2f} | {cmp.baseline_ci.mean:.2f} | "
                f"[{ci.low:.2f}, {ci.high:.2f}] | {cmp.p_value:.4f} | {'YES' if sig else 'no'} | "
                f"{cmp.cohens_d:.2f} |"
            )
    lines += [
        "",
        "## Reproduce",
        "",
        "```bash",
        f"python scripts/run_city_evaluation.py --city {settings['city']} --scale {settings['scale']} "
        f"--seeds {settings['seeds']} --decisions {settings['decisions']}",
        "```",
        "",
        "Per-cell raw metrics are in `data/networks/<city>/runs/evaluation_cells.jsonl`.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--city", default="manhattan")
    parser.add_argument("--period", default="pm_peak")
    parser.add_argument("--scale", type=float, default=0.25)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--decisions", type=int, default=180)
    parser.add_argument("--teleport-s", type=float, default=300)
    parser.add_argument(
        "--max-wall-s", type=float, default=0, help="stop starting new cells after this"
    )
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    city_dir = project_root() / "data" / "networks" / args.city
    net = city_dir / "net.net.xml"
    route = city_dir / "demand" / f"{args.period}.rou.xml"
    vtypes = city_dir / "vtypes.add.xml"
    runs_dir = city_dir / "runs"
    runs_dir.mkdir(exist_ok=True)
    cells_path = runs_dir / "evaluation_cells.jsonl"
    tripinfo_dir = runs_dir / "tripinfo"
    tripinfo_dir.mkdir(exist_ok=True)

    index = NetworkSignalIndex.from_net_file(net)
    tls_ids = index.controllable_ids
    settings = {
        "city": args.city,
        "signals": len(tls_ids),
        "scale": args.scale,
        "decisions": args.decisions,
        "seeds": args.seeds,
        "teleport_s": args.teleport_s,
    }

    cells = load_cells(cells_path)
    started = time.monotonic()
    if not args.report_only:
        done = {(str(c["controller"]), int(c["seed"])) for c in cells}  # type: ignore[call-overload]
        # Seed-major order: every controller gets seed 1 before anyone gets seed 2, so a run cut
        # short still compares like with like.
        for seed in range(1, args.seeds + 1):
            for controller in CONTROLLERS:
                if (controller, seed) in done:
                    continue
                if args.max_wall_s and time.monotonic() - started > args.max_wall_s:
                    logger.info("city_eval.wall_budget_reached")
                    break
                cell_started = time.monotonic()
                metrics = run_scenario_cell(
                    mode=controller,  # type: ignore[arg-type]
                    tls_ids=tls_ids,
                    net_file=net,
                    route_file=route,
                    seed=seed,
                    num_decisions=args.decisions,
                    tripinfo_output=tripinfo_dir / f"{controller}_seed{seed}.xml",
                    additional_files=[vtypes] if vtypes.is_file() else [],
                    max_lanes=max(10, index.required_max_lanes()),
                    max_green_phases=max(2, index.required_max_green_phases()),
                    programs=index.programs(),
                    native_durations={t: index.native_durations(t) for t in tls_ids},
                    time_to_teleport_s=args.teleport_s,
                    scale=args.scale,
                    queue_sample_every=6,
                )
                record = {
                    "controller": controller,
                    "seed": seed,
                    "scale": args.scale,
                    "decisions": args.decisions,
                    "wall_s": round(time.monotonic() - cell_started, 1),
                    **asdict(metrics),
                }
                with cells_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
                cells.append(record)
                print(json.dumps(record), flush=True)

    report = build_report(cells, settings)
    out = project_root() / "docs" / "report" / f"city_evaluation_{args.city}.md"
    out.write_text(report, encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
