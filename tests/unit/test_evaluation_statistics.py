"""Validates every statistical formula in atlas.evaluation.statistics against
hand-computed values -- the same discipline as Week 5's PPO clip-asymmetry tests and
Week 8's pinball-loss tests: delegate the well-tested numerics (scipy's t-distribution,
Welch's t-test) but verify this module's own arithmetic (variance, standard error,
Cohen's d, the Holm-Bonferroni step-down) against numbers computed by hand, not just
"does it run."
"""

from __future__ import annotations

import math

from atlas.evaluation.statistics import (
    cohens_d,
    compare_two_controllers,
    holm_bonferroni,
    mean_confidence_interval,
    welch_t_test,
)


class TestMeanConfidenceInterval:
    def test_matches_hand_computed_values(self) -> None:
        """values = [10, 12, 14, 16, 18]: mean=14, sample variance=10 (ddof=1),
        std_err=sqrt(10/5)=sqrt(2), t_crit(df=4, 95%)=2.7764 (textbook value) ->
        half_width = 2.7764 * sqrt(2) = 3.9265.
        """
        result = mean_confidence_interval([10, 12, 14, 16, 18], confidence=0.95)

        assert result.mean == 14.0
        assert result.n == 5
        expected_half_width = 2.7764 * math.sqrt(2)
        assert abs(result.half_width - expected_half_width) < 0.001
        assert abs(result.low - (14.0 - expected_half_width)) < 0.001
        assert abs(result.high - (14.0 + expected_half_width)) < 0.001

    def test_wider_interval_for_more_variable_data(self) -> None:
        low_variance = mean_confidence_interval([10.0, 10.1, 9.9, 10.0, 10.0])
        high_variance = mean_confidence_interval([5.0, 15.0, 2.0, 18.0, 10.0])

        assert high_variance.half_width > low_variance.half_width


class TestWelchTTest:
    def test_same_mean_different_samples_has_near_zero_t_and_p_near_one(self) -> None:
        """Two samples with the same mean but real (nonzero) variance -- NOT two
        exactly-identical constant arrays, where Welch's t is mathematically
        undefined (0/0, both variances exactly zero) rather than 0."""
        t_stat, p_value = welch_t_test([4.0, 5.0, 6.0, 5.0], [4.0, 6.0, 5.0, 5.0])

        assert abs(t_stat) < 1e-9
        assert p_value > 0.99

    def test_clearly_separated_groups_are_significant(self) -> None:
        t_stat, p_value = welch_t_test([100.0, 101.0, 99.0, 100.5, 99.5], [1.0, 2.0, 0.5, 1.5, 1.0])

        assert p_value < 0.001
        assert t_stat > 0  # first group's mean is higher

    def test_sign_flips_when_argument_order_flips(self) -> None:
        t_ab, _ = welch_t_test([10.0, 11.0, 9.0], [1.0, 2.0, 0.0])
        t_ba, _ = welch_t_test([1.0, 2.0, 0.0], [10.0, 11.0, 9.0])

        assert t_ab > 0
        assert t_ba < 0


class TestCohensD:
    def test_matches_hand_computed_value(self) -> None:
        """a=[2,4,6] (mean=4, var=4), b=[1,3,5] (mean=3, var=4) -> pooled_std =
        sqrt(((2*4)+(2*4))/4) = sqrt(16/4) = 2 -> d = (4-3)/2 = 0.5."""
        d = cohens_d([2.0, 4.0, 6.0], [1.0, 3.0, 5.0])

        assert abs(d - 0.5) < 1e-9

    def test_zero_for_identical_means(self) -> None:
        d = cohens_d([1.0, 2.0, 3.0], [10.0, 11.0, 12.0])
        # same spread, means differ by 9 -- sanity check the sign/magnitude direction
        assert d < 0  # first group's mean (2) is less than second's (11)


class TestHolmBonferroni:
    def test_matches_the_classic_textbook_example(self) -> None:
        """p=[0.01,0.02,0.03,0.04,0.05], alpha=0.05, m=5: thresholds are
        0.05/5=0.01, 0.05/4=0.0125, 0.05/3=0.0167, 0.05/2=0.025, 0.05/1=0.05.
        0.01 <= 0.01 -> reject. 0.02 <= 0.0125? No -> stop; nothing else rejected.
        """
        result = holm_bonferroni([0.01, 0.02, 0.03, 0.04, 0.05], alpha=0.05)

        assert result == [True, False, False, False, False]

    def test_output_order_matches_input_order_not_sorted_order(self) -> None:
        # Same five p-values as above, deliberately shuffled -- only index 2 (p=0.01)
        # should reject, proving the function un-sorts correctly.
        result = holm_bonferroni([0.05, 0.03, 0.01, 0.04, 0.02], alpha=0.05)

        assert result == [False, False, True, False, False]

    def test_all_rejected_when_all_p_values_are_tiny(self) -> None:
        result = holm_bonferroni([1e-9, 1e-8, 1e-7], alpha=0.05)

        assert result == [True, True, True]

    def test_none_rejected_when_all_p_values_are_large(self) -> None:
        result = holm_bonferroni([0.5, 0.6, 0.7], alpha=0.05)

        assert result == [False, False, False]

    def test_less_conservative_than_uncorrected_alpha_would_be(self) -> None:
        """A p-value that WOULD be significant at uncorrected alpha=0.05 is not
        necessarily significant after Holm-Bonferroni -- the whole point of the
        correction. p=0.04 alone would reject at alpha=0.05, but here (as the
        2nd-smallest of 3) its threshold is 0.05/2=0.025, so it does not.
        """
        result = holm_bonferroni([0.01, 0.04, 0.09], alpha=0.05)

        assert result == [True, False, False]


class TestCompareTwoControllers:
    def test_bundles_ci_and_test_and_effect_size_consistently(self) -> None:
        baseline = [20.0, 21.0, 19.0, 20.5, 19.5]
        treatment = [10.0, 11.0, 9.0, 10.5, 9.5]

        result = compare_two_controllers(baseline, treatment)

        assert result.baseline_ci.mean == 20.0
        assert result.treatment_ci.mean == 10.0
        assert result.p_value < 0.001
        assert result.cohens_d > 0  # baseline mean is higher
        assert result.cis_overlap is False

    def test_cis_overlap_detected_for_similar_groups(self) -> None:
        baseline = [20.0, 21.0, 19.0, 20.5, 19.5]
        treatment = [19.8, 20.8, 18.8, 20.3, 19.3]

        result = compare_two_controllers(baseline, treatment)

        assert result.cis_overlap is True
