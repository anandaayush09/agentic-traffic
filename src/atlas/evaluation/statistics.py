"""The statistical protocol (docs/design/11_evaluation.md section 11.3): "Report mean
+/- 95% CI (t-distribution); controller comparisons via Welch's t-test with
Holm-Bonferroni correction across scenarios; effect sizes (Cohen's d) alongside
p-values." This module is pure math with no SUMO/twin dependency, by design -- the
same separation `clipped_policy_loss`/`clipped_value_loss` (Week 5's PPO) got, so
every formula here is directly testable against hand-computed values, not only
exercised indirectly through a full evaluation run.

Deliberately uses `scipy.stats` for the t-distribution/Welch's-t-test primitives
themselves (a from-scratch statistics library would be reinventing well-tested,
easy-to-get-subtly-wrong numerics for no portfolio value -- unlike MAPPO, which this
project *does* build from scratch because that's the point of the exercise), but
Holm-Bonferroni correction is hand-implemented: it's a short, simple step-down
procedure with no scipy equivalent that doesn't pull in a heavier stats package
(statsmodels) for one function.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy import stats


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    mean: float
    low: float
    high: float
    confidence: float
    n: int

    @property
    def half_width(self) -> float:
        return self.high - self.mean


def mean_confidence_interval(values: list[float], confidence: float = 0.95) -> ConfidenceInterval:
    """Mean +/- a t-distribution confidence interval -- correct for the SMALL sample
    sizes this project's 10-seeds-per-cell protocol actually has (a normal-
    distribution/z-score interval understates uncertainty at n=10; the t-distribution
    is wider precisely because it accounts for that), per docs/design/11_evaluation.md
    section 11.3's own explicit choice.
    """
    if len(values) < 2:
        raise ValueError("Need at least 2 values to compute a confidence interval")
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)  # sample variance, ddof=1
    std_err = math.sqrt(variance / n)
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    half_width = t_crit * std_err
    return ConfidenceInterval(
        mean=mean, low=mean - half_width, high=mean + half_width, confidence=confidence, n=n
    )


def welch_t_test(a: list[float], b: list[float]) -> tuple[float, float]:
    """Welch's t-test (unequal-variance t-test) -- the correct choice here since
    different controllers' run-to-run variance is not assumed equal (max-pressure's
    determinism vs. a stochastic RL policy's variance, concretely). Returns
    (t_statistic, p_value), two-sided.
    """
    result = stats.ttest_ind(a, b, equal_var=False)
    return float(result.statistic), float(result.pvalue)


def cohens_d(a: list[float], b: list[float]) -> float:
    """Standard pooled-variance Cohen's d effect size: (mean_a - mean_b) / pooled_std.
    Reported alongside the p-value (docs/design/11_evaluation.md section 11.3) because
    a p-value alone conflates "the effect is real" with "the effect is large" --
    a large-n comparison can have p < 0.001 for a practically meaningless difference.
    """
    n_a, n_b = len(a), len(b)
    mean_a, mean_b = sum(a) / n_a, sum(b) / n_b
    var_a = sum((v - mean_a) ** 2 for v in a) / (n_a - 1)
    var_b = sum((v - mean_b) ** 2 for v in b) / (n_b - 1)
    pooled_std = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    return (mean_a - mean_b) / pooled_std


def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Holm-Bonferroni step-down correction for multiple comparisons (controlling the
    family-wise error rate across every scenario a controller is compared on, per
    docs/design/11_evaluation.md section 11.3 -- without this, testing e.g. 10
    scenarios at alpha=0.05 each gives a much higher than 5% chance of at least one
    false positive across the family).

    Algorithm (Holm 1979): sort p-values ascending; for the i-th smallest (1-indexed)
    of m total, reject its null hypothesis only if p_i <= alpha / (m - i + 1) AND
    every smaller p-value was also rejected -- the step-down stops at the first
    non-rejection. Returns a list of reject/not-reject booleans in the SAME order as
    the input `p_values`, not sorted order.
    """
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    reject = [False] * m
    for rank, idx in enumerate(order):  # rank is 0-indexed; i = rank + 1
        threshold = alpha / (m - rank)
        if p_values[idx] <= threshold:
            reject[idx] = True
        else:
            break  # step-down: first non-rejection stops the whole procedure
    return reject


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    baseline_ci: ConfidenceInterval
    treatment_ci: ConfidenceInterval
    t_statistic: float
    p_value: float
    cohens_d: float

    @property
    def cis_overlap(self) -> bool:
        return not (
            self.treatment_ci.high < self.baseline_ci.low
            or self.baseline_ci.high < self.treatment_ci.low
        )


def compare_two_controllers(baseline: list[float], treatment: list[float]) -> ComparisonResult:
    """Bundles everything one claims-table row needs for one (scenario, baseline,
    treatment) comparison: both controllers' CIs, Welch's t-test, and Cohen's d.
    `holm_bonferroni` is applied separately, across the p-values of every row in a
    claims table at once -- it needs the whole family of comparisons, not one pair.
    """
    t_statistic, p_value = welch_t_test(baseline, treatment)
    return ComparisonResult(
        baseline_ci=mean_confidence_interval(baseline),
        treatment_ci=mean_confidence_interval(treatment),
        t_statistic=t_statistic,
        p_value=p_value,
        cohens_d=cohens_d(baseline, treatment),
    )
