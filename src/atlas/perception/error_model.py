"""Fits the perception error model (docs/design/06_ai_pipeline.md section 6.3,
deliverable 3: "per-condition (day/night, occlusion density) miss rate, FP rate, count
MAE -- fitted as the twin's sensor-noise parameters"). Scoped like Webster's formula and
green-wave offsets before it: the fitting statistics themselves, verified by hand here;
wiring the resulting per-condition rates into the twin's own sensor-noise model (so a
trained control policy sees realistically-degraded observations, the actual "robustness
bridge" R7 in docs/design/12_risks.md) is a separate, later integration, not assumed
done by this module.

Inputs are per-clip detection-vs-ground-truth tallies, not raw detections -- this module
doesn't run a detector or compute IoU-based matching itself (that's YOLO fine-tuning +
evaluation, docs/design/06_ai_pipeline.md section 6.3's other deliverables); it fits
rates from whatever tally a detector evaluation already produced.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DetectionTally:
    """One clip's detection-vs-ground-truth counts under one labeled condition
    (e.g. "day_low_occlusion", "night_high_occlusion") -- the unit `fit_error_model`
    aggregates over.
    """

    condition: str
    ground_truth_count: int
    detected_count: int
    true_positives: int
    false_positives: int
    false_negatives: int


@dataclass(frozen=True, slots=True)
class ErrorModelStats:
    condition: str
    miss_rate: float
    """false_negatives / (true_positives + false_negatives) -- fraction of real
    vehicles the detector missed entirely, pooled across every clip in this condition."""
    fp_rate: float
    """false_positives / (true_positives + false_positives) -- fraction of reported
    detections that weren't real vehicles."""
    count_mae: float
    """Mean absolute error between detected_count and ground_truth_count, averaged
    per-clip (not pooled) -- the count-level error a virtual-loop consumer actually
    experiences, distinct from miss_rate/fp_rate's per-detection view."""
    num_clips: int


def fit_error_model(tallies: Sequence[DetectionTally]) -> dict[str, ErrorModelStats]:
    if not tallies:
        raise ValueError("tallies must be non-empty")

    grouped: dict[str, list[DetectionTally]] = defaultdict(list)
    for tally in tallies:
        grouped[tally.condition].append(tally)

    return {condition: _fit_one_condition(condition, group) for condition, group in grouped.items()}


def _fit_one_condition(condition: str, tallies: list[DetectionTally]) -> ErrorModelStats:
    total_tp = sum(t.true_positives for t in tallies)
    total_fp = sum(t.false_positives for t in tallies)
    total_fn = sum(t.false_negatives for t in tallies)

    miss_rate = total_fn / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    fp_rate = total_fp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    count_mae = sum(abs(t.detected_count - t.ground_truth_count) for t in tallies) / len(tallies)

    return ErrorModelStats(
        condition=condition,
        miss_rate=miss_rate,
        fp_rate=fp_rate,
        count_mae=count_mae,
        num_clips=len(tallies),
    )
