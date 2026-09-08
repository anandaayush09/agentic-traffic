#!/usr/bin/env python
"""Week 11's detector-dropout robustness sweep (docs/design/11_evaluation.md section
11.5.2: "Detector dropout: 0-50% of detectors dead; success criterion: never worse
than actuated (per NFR)").

Runs max-pressure and actuated (the two queue-reading controllers -- fixed-time is
unaffected by construction, see atlas.evaluation.runner's own note) at detector
dropout rates 0%, 25%, 50% against the real manhattan_midtown corridor, 10 seeds
each, and checks the design's literal success criterion: does max-pressure's mean
waiting time under dropout stay <= actuated's own mean waiting time at the same
dropout rate? (Not "no degradation at all" -- SOME degradation under real detector
loss is expected and honest; the NFR is a relative floor against a detector-
independent-ish alternative, not an absolute one.)

Perception noise sweep (11.5.1) and hint dropout (11.5.3) are NOT run here -- both
are fundamentally about a trained RL policy's degradation (the design's own
hypothesis for 11.5.1 is literally "RL trained with noise randomization holds a
margin over max-pressure"; corridor hints, 11.5.3, only ever feed the RL
observation), and no converged MAPPO checkpoint exists in this environment (Week 6's
own documented stance, still true here) -- real, named gaps, not attempted with a
placeholder.

Usage:
    python scripts/run_robustness_sweep.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atlas.common.config import project_root  # noqa: E402
from atlas.common.logging import configure_logging, get_logger  # noqa: E402
from atlas.evaluation.runner import run_scenario_cell  # noqa: E402
from atlas.evaluation.statistics import compare_two_controllers, holm_bonferroni  # noqa: E402

configure_logging()
logger = get_logger(__name__)

CORRIDOR_TLS_IDS = ["42430333", "42430338", "42430342", "42430344", "42430347"]
NUM_SEEDS = 10
NUM_DECISIONS = 200  # 1000 simulated seconds -- no scenario window to cover here,
# so a shorter episode than the pilot report's is enough for a real, fast sweep.
DROPOUT_RATES = [0.0, 0.25, 0.5]
CONTROLLERS = ["max_pressure", "actuated"]


def run_sweep() -> dict[float, dict[str, list[float]]]:
    """Returns dropout_rate -> controller -> [mean_waiting_time_s per seed]."""
    network_dir = project_root() / "data" / "networks" / "manhattan_midtown"
    net_file = network_dir / "net.net.xml"
    route_file = network_dir / "demand" / "pm_peak.rou.xml"
    vtypes_file = network_dir / "vtypes.add.xml"
    tripinfo_dir = project_root() / "data" / "robustness_sweep_tripinfo"
    tripinfo_dir.mkdir(parents=True, exist_ok=True)

    results: dict[float, dict[str, list[float]]] = {}
    for dropout_rate in DROPOUT_RATES:
        results[dropout_rate] = {}
        for controller in CONTROLLERS:
            waiting_times = []
            for seed in range(1, NUM_SEEDS + 1):
                tripinfo_output = (
                    tripinfo_dir / f"dropout{dropout_rate}_{controller}_seed{seed}.xml"
                )
                metrics = run_scenario_cell(
                    mode=controller,  # type: ignore[arg-type]
                    tls_ids=CORRIDOR_TLS_IDS,
                    net_file=net_file,
                    route_file=route_file,
                    seed=seed,
                    num_decisions=NUM_DECISIONS,
                    tripinfo_output=tripinfo_output,
                    additional_files=[vtypes_file],
                    detector_dropout_rate=dropout_rate,
                )
                waiting_times.append(metrics.mean_waiting_time_s)
                logger.info(
                    "robustness_sweep.cell_complete",
                    dropout_rate=dropout_rate,
                    controller=controller,
                    seed=seed,
                    mean_waiting_time_s=metrics.mean_waiting_time_s,
                )
            results[dropout_rate][controller] = waiting_times
    return results


def report(results: dict[float, dict[str, list[float]]]) -> None:
    """A raw mean comparison alone would be misleading here -- max-pressure and
    actuated's means are close enough at low dropout that "which is bigger" can flip
    on noise alone. Uses the same statistical protocol (docs/design/11_evaluation.md
    section 11.3) as the main pilot report: a raw mean difference only counts as a
    real NFR violation if it's actually statistically significant, corrected for
    testing all 3 dropout rates at once (Holm-Bonferroni) -- not raw greater-than.
    """
    comparisons = [
        (
            dropout_rate,
            compare_two_controllers(
                results[dropout_rate]["actuated"], results[dropout_rate]["max_pressure"]
            ),
        )
        for dropout_rate in DROPOUT_RATES
    ]
    significant = holm_bonferroni([c.p_value for _, c in comparisons], alpha=0.05)

    print("\n=== Detector dropout robustness sweep (mean waiting time, s) ===")
    print(
        f"{'dropout':>10} | {'max_pressure':>14} | {'actuated':>10} | {'p-value':>9} | "
        "significantly worse than actuated?"
    )
    any_real_violation = False
    for (dropout_rate, comparison), is_significant in zip(comparisons, significant, strict=True):
        mp_mean = comparison.treatment_ci.mean
        act_mean = comparison.baseline_ci.mean
        # "significantly worse" needs BOTH a higher raw mean AND a statistically
        # significant difference -- a significant difference in max-pressure's FAVOR
        # is not an NFR violation.
        is_violation = is_significant and mp_mean > act_mean
        any_real_violation = any_real_violation or is_violation
        print(
            f"{dropout_rate:>10.0%} | {mp_mean:>14.2f} | {act_mean:>10.2f} | "
            f"{comparison.p_value:>9.4f} | {'YES -- violation' if is_violation else 'no'}"
        )
    print(
        f"\nOverall NFR (never SIGNIFICANTLY worse than actuated, any dropout rate): "
        f"{'FAIL' if any_real_violation else 'PASS'}"
    )


def main() -> None:
    results = run_sweep()
    report(results)


if __name__ == "__main__":
    main()
