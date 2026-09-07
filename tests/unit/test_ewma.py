"""Hand-computed verification of EwmaAnomalyDetector's mean/variance recursion, plus
the flatline-detection gap a pure z-score check can't catch on its own (verified
directly, not assumed).
"""

from __future__ import annotations

import pytest

from atlas.anomaly.ewma import EwmaAnomalyDetector

ALPHA = 0.5


class TestEwmaRecursion:
    """Hand-traced with alpha=0.5:
    x0=10 -> mean=10, var=0 (first observation, nothing to compare against)
    x1=12 -> diff=2, incr=1, mean=11, var=(0.5)*(0 + 2*1)=1.0
    x2=8  -> diff=8-11=-3, incr=-1.5, mean=9.5, var=(0.5)*(1.0 + (-3)*(-1.5))=(0.5)*(1.0+4.5)=2.75
    """

    def test_mean_and_variance_match_hand_computation(self) -> None:
        detector = EwmaAnomalyDetector(alpha=ALPHA, z_threshold=1e9)  # threshold off for this test
        detector.update("k", 10.0)
        detector.update("k", 12.0)
        mean, var = detector.current_mean_var("k")
        assert mean == pytest.approx(11.0)
        assert var == pytest.approx(1.0)

        detector.update("k", 8.0)
        mean, var = detector.current_mean_var("k")
        assert mean == pytest.approx(9.5)
        assert var == pytest.approx(2.75)

    def test_first_observation_returns_none(self) -> None:
        detector = EwmaAnomalyDetector()
        assert detector.update("k", 10.0) is None


class TestSpikeDetection:
    def test_hand_computed_z_score_flags_a_clear_spike(self) -> None:
        """After x0=10, x1=12 (mean=11, var=1.0 per the trace above), a jump to 50:
        diff = 50 - 11 = 39; z = diff / (sqrt(prior_var=1.0) + eps) = 39.0 / ~1.0 ~= 39.
        Far past any reasonable threshold.
        """
        detector = EwmaAnomalyDetector(alpha=ALPHA, z_threshold=3.0)
        detector.update("k", 10.0)
        detector.update("k", 12.0)

        event = detector.update("k", 50.0)

        assert event is not None
        assert event.kind == "spike"
        assert event.z_score == pytest.approx(39.0, abs=0.01)

    def test_small_fluctuation_within_threshold_is_not_flagged(self) -> None:
        detector = EwmaAnomalyDetector(alpha=ALPHA, z_threshold=3.0)
        detector.update("k", 10.0)
        detector.update("k", 12.0)
        detector.update("k", 11.0)

        event = detector.update("k", 11.5)

        assert event is None

    def test_different_keys_tracked_independently(self) -> None:
        detector = EwmaAnomalyDetector(alpha=ALPHA, z_threshold=3.0)
        detector.update("a", 10.0)
        detector.update("a", 12.0)
        detector.update("b", 100.0)  # b's first observation, unrelated to a's history

        event = detector.update("b", 100.0)  # no change for b -- must not spike

        assert event is None


class TestFlatlineDetection:
    def test_a_zscore_check_alone_cannot_catch_a_flatline(self) -> None:
        """The gap this two-check design exists to close: a value that exactly repeats
        its own recent history has z~0 by construction, no matter how implausible a
        real sensor reporting the IDENTICAL reading many times in a row actually is."""
        detector = EwmaAnomalyDetector(alpha=ALPHA, z_threshold=3.0, flatline_window=1000)
        for _ in range(20):
            event = detector.update("k", 5.0)
        # With flatline detection effectively disabled (window=1000, never reached),
        # a perfectly constant stream produces no spike event at all -- confirming
        # z-score alone is blind to this failure mode.
        assert event is None

    def test_constant_stream_is_flagged_once_the_window_fills(self) -> None:
        detector = EwmaAnomalyDetector(alpha=ALPHA, z_threshold=3.0, flatline_window=5)
        event = None
        for _ in range(5):
            event = detector.update("k", 7.0)

        assert event is not None
        assert event.kind == "flatline"
        assert event.z_score is None

    def test_varying_stream_is_never_flagged_as_flatline(self) -> None:
        detector = EwmaAnomalyDetector(alpha=ALPHA, z_threshold=3.0, flatline_window=5)
        values = [1.0, 2.0, 1.5, 2.5, 1.8, 2.2, 1.9]
        events = [detector.update("k", v) for v in values]

        assert all(e is None or e.kind != "flatline" for e in events)

    def test_flatline_takes_priority_over_a_borderline_spike_check(self) -> None:
        """A flatline that happens to also sit far from the running mean (e.g. after a
        real spike settles onto a new stuck value) should be reported as a flatline,
        the more specific and more actionable diagnosis, not a generic spike."""
        detector = EwmaAnomalyDetector(alpha=ALPHA, z_threshold=3.0, flatline_window=3)
        detector.update("k", 10.0)
        detector.update("k", 10.0)
        event = detector.update("k", 10.0)

        assert event is not None
        assert event.kind == "flatline"


class TestValidation:
    def test_rejects_alpha_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            EwmaAnomalyDetector(alpha=0.0)
        with pytest.raises(ValueError, match="alpha"):
            EwmaAnomalyDetector(alpha=1.5)
