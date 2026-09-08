#!/usr/bin/env python
"""Week 11's evaluation report, generated programmatically from real runs
(docs/design/11_evaluation.md; docs/design/13_roadmap.md's Week 11 deliverable:
"docs/report/ -- the full evaluation report with every claim traceable").

Scope, deliberately and stated up front: the design's full suite is "~35-cell grid x
4 controllers x 10 seeds ~= 1,400 evaluation runs... a few days of background
compute" (section 11.4's own words) -- not something to run inside an interactive
session. This script runs a REAL, honest PILOT subset instead: 3 controllers
(max-pressure, fixed-time, actuated -- RL excluded, see runner.py's own scope note)
x 5 conditions (no disruption, plus the 4 real scenario types built so far:
accident/closure/construction/weather) x 5 seeds x 360 real decision cycles (1800
simulated seconds) against the real manhattan_midtown corridor -- every number in
the generated report is a real measurement, none fabricated or interpolated, but the
sample size and episode length are smaller than the design's full-scale numbers.
Scaling up is changing NUM_SEEDS/NUM_DECISIONS/the scenario grid here and re-running;
the statistical machinery (compare_two_controllers, holm_bonferroni) is identical at
any scale.

Usage:
    python scripts/run_evaluation_pilot.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atlas.common.config import project_root  # noqa: E402
from atlas.common.logging import configure_logging, get_logger  # noqa: E402
from atlas.evaluation.runner import run_scenario_cell  # noqa: E402
from atlas.evaluation.statistics import compare_two_controllers, holm_bonferroni  # noqa: E402
from atlas.twin.scenario import ScenarioSpec  # noqa: E402

configure_logging()
logger = get_logger(__name__)

CORRIDOR_TLS_IDS = ["42430333", "42430338", "42430342", "42430344", "42430347"]
NUM_SEEDS = 10  # docs/design/11_evaluation.md section 11.3's own spec -- a real
# first pilot run at 5 seeds showed large, consistent effect sizes (Cohen's d ~2.3-2.7
# for actuated vs fixed-time) that still didn't survive Holm-Bonferroni correction
# across all 10 comparisons at n=5; this is exactly the statistical-power argument
# for why the design specifies 10 seeds, not a smaller pilot-sized number.
NUM_DECISIONS = 360  # 1800 simulated seconds -- long enough for every scenario
# below's own [start_s, start_s+duration_s) window (the latest, weather, ends at
# 1800s) to actually trigger and complete within the episode.
CONTROLLERS = ["max_pressure", "fixed_time", "actuated"]
BASELINE_CONTROLLER = "fixed_time"  # the design's own "floor every system must beat"

CONDITIONS: dict[str, ScenarioSpec | None] = {
    "no_disruption": None,
    "accident": ScenarioSpec(
        name="midblock_accident",
        type="accident",
        edge_id="448686048#0",
        start_s=600,
        duration_s=300,
        speed_factor=0.25,
    ),
    "closure": ScenarioSpec(
        name="avenue_closure", type="closure", edge_id="1320232776#0", start_s=900, duration_s=300
    ),
    "construction": ScenarioSpec(
        name="lane_drop_construction",
        type="construction",
        edge_id="448686048#0",
        start_s=1200,
        duration_s=300,
        speed_factor=0.7,
        lane_indices=[0, 1],
    ),
    "weather": ScenarioSpec(
        name="regional_weather",
        type="weather",
        edge_id="1320232776#0",
        area_edge_ids=["448686048#0"],
        start_s=1500,
        duration_s=300,
        speed_factor=0.6,
    ),
}


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001 -- report generation must not fail just because
        # git isn't available in whatever environment eventually runs this.
        return "unknown"


def run_pilot() -> dict[str, dict[str, list[float]]]:
    """Returns condition -> controller -> [mean_waiting_time_s per seed]."""
    network_dir = project_root() / "data" / "networks" / "manhattan_midtown"
    net_file = network_dir / "net.net.xml"
    route_file = network_dir / "demand" / "pm_peak.rou.xml"
    vtypes_file = network_dir / "vtypes.add.xml"
    tripinfo_dir = project_root() / "data" / "eval_pilot_tripinfo"
    tripinfo_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, list[float]]] = {}
    for condition_name, spec in CONDITIONS.items():
        results[condition_name] = {}
        for controller in CONTROLLERS:
            waiting_times = []
            for seed in range(1, NUM_SEEDS + 1):
                tripinfo_output = tripinfo_dir / f"{condition_name}_{controller}_seed{seed}.xml"
                metrics = run_scenario_cell(
                    mode=controller,  # type: ignore[arg-type]
                    tls_ids=CORRIDOR_TLS_IDS,
                    net_file=net_file,
                    route_file=route_file,
                    seed=seed,
                    num_decisions=NUM_DECISIONS,
                    tripinfo_output=tripinfo_output,
                    additional_files=[vtypes_file],
                    scenario_specs=[spec] if spec is not None else None,
                )
                waiting_times.append(metrics.mean_waiting_time_s)
                logger.info(
                    "eval_pilot.cell_complete",
                    condition=condition_name,
                    controller=controller,
                    seed=seed,
                    mean_waiting_time_s=metrics.mean_waiting_time_s,
                )
            results[condition_name][controller] = waiting_times
    return results


def generate_report(results: dict[str, dict[str, list[float]]]) -> str:
    rows = []
    p_values = []
    for condition_name in CONDITIONS:
        baseline = results[condition_name][BASELINE_CONTROLLER]
        for controller in CONTROLLERS:
            if controller == BASELINE_CONTROLLER:
                continue
            comparison = compare_two_controllers(baseline, results[condition_name][controller])
            p_values.append(comparison.p_value)
            rows.append((condition_name, controller, comparison))

    significant = holm_bonferroni(p_values, alpha=0.05)

    lines = [
        "# Week 11 Evaluation Pilot Report",
        "",
        f"Git SHA: `{_git_sha()}`",
        f"Network: manhattan_midtown corridor ({', '.join(CORRIDOR_TLS_IDS)})",
        f"Seeds per cell: {NUM_SEEDS} (seeds 1-{NUM_SEEDS})",
        f"Decisions per episode: {NUM_DECISIONS} ({NUM_DECISIONS * 5}s simulated)",
        f"Baseline controller: {BASELINE_CONTROLLER}",
        "",
        "**Scope**: a real, honest pilot subset, not the full ~1,400-run suite "
        "docs/design/11_evaluation.md section 11.4 specifies -- see this script's "
        "own module docstring for what scaling up means and why this size was chosen.",
        "",
        "## Claims table",
        "",
        "| Condition | Controller vs. baseline | Mean waiting time (s) | Baseline mean (s) | "
        "95% CI (treatment) | Welch's t | p-value | Holm-Bonferroni significant? | Cohen's d |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for (condition_name, controller, comparison), is_significant in zip(
        rows, significant, strict=True
    ):
        ci = comparison.treatment_ci
        lines.append(
            f"| {condition_name} | {controller} vs {BASELINE_CONTROLLER} | "
            f"{ci.mean:.2f} | {comparison.baseline_ci.mean:.2f} | "
            f"[{ci.low:.2f}, {ci.high:.2f}] | {comparison.t_statistic:.3f} | "
            f"{comparison.p_value:.4f} | {'YES' if is_significant else 'no'} | "
            f"{comparison.cohens_d:.3f} |"
        )

    lines += [
        "",
        "## How to reproduce",
        "",
        "```bash",
        "python scripts/run_evaluation_pilot.py",
        "```",
        "",
        "Deterministic given the same git SHA and seeds -- rerunning regenerates an "
        "identical claims table (docs/design/13_roadmap.md's Week 11 test: "
        '"report-generation reproducibility").',
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    results = run_pilot()
    report = generate_report(results)
    output_path = project_root() / "docs" / "report" / "evaluation_pilot_report.md"
    output_path.write_text(report)
    print(f"Wrote report to {output_path}")
    print()
    print(report)


if __name__ == "__main__":
    main()
