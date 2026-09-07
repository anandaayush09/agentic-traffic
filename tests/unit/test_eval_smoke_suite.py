"""Validates ScenarioResult's pass/fail logic -- the actual regression-gate decision
scripts/eval_smoke_suite.py's CI job lives or dies by. Doesn't re-run the scenarios
themselves (that needs a real network build + SUMO twin, exercised by running the script
directly, e.g. in .github/workflows/eval.yml) -- just the comparison rule.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from eval_smoke_suite import ScenarioResult  # noqa: E402


class TestScenarioResultPassed:
    def test_passes_when_max_pressure_beats_fixed_time(self) -> None:
        result = ScenarioResult("s", max_pressure_mean=5.0, fixed_time_mean=10.0)
        assert result.passed is True

    def test_passes_on_exact_tie(self) -> None:
        result = ScenarioResult("s", max_pressure_mean=5.0, fixed_time_mean=5.0)
        assert result.passed is True

    def test_fails_when_max_pressure_is_worse(self) -> None:
        """The actual regression this gate exists to catch: something broke max-
        pressure's control law, and it's now worse than the naive fixed-time fallback."""
        result = ScenarioResult("s", max_pressure_mean=10.0, fixed_time_mean=5.0)
        assert result.passed is False
