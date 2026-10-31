"""Validates the evaluation report's optional RL arm on synthetic numbers (the real runs are
far too long for a unit test): the RL controller is compared to the same baseline with the same
statistics, the report names which checkpoint it used, and the classical-only report is unchanged.
"""

from __future__ import annotations

import random

import run_evaluation_pilot as pilot


def _results(controllers: list[str]) -> dict[str, dict[str, list[float]]]:
    rng = random.Random(7)
    means = {"fixed_time": 170.0, "max_pressure": 160.0, "actuated": 150.0, "rl": 140.0}
    return {
        condition: {c: [rng.gauss(means[c], 5.0) for _ in range(10)] for c in controllers}
        for condition in pilot.CONDITIONS
    }


class TestReportWithAnRlArm:
    def test_rl_is_compared_to_the_baseline_and_the_checkpoint_is_named(self) -> None:
        controllers = [*pilot.CONTROLLERS, "rl"]

        report = pilot.generate_report(
            _results(controllers),
            controllers,
            "mappo_district.pt (iteration 120; trained over 22 junctions)",
        )

        assert "rl vs fixed_time" in report
        assert "RL policy: mappo_district.pt (iteration 120" in report
        # 5 conditions x 3 non-baseline controllers
        assert report.count(" vs fixed_time |") == 15

    def test_the_classical_only_report_is_unchanged(self) -> None:
        report = pilot.generate_report(_results(pilot.CONTROLLERS))

        assert "rl vs fixed_time" not in report
        assert "RL policy" not in report
        assert report.count(" vs fixed_time |") == 10

    def test_holm_bonferroni_now_covers_the_extra_comparisons(self) -> None:
        controllers = [*pilot.CONTROLLERS, "rl"]

        report = pilot.generate_report(_results(controllers), controllers, "x.pt")

        # Every row carries a significance verdict, including the new RL rows.
        rl_rows = [line for line in report.splitlines() if "| rl vs fixed_time |" in line]
        assert len(rl_rows) == 5
        assert all(("YES" in row) or ("no" in row) for row in rl_rows)
