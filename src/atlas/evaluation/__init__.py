"""The evaluation plane (docs/design/11_evaluation.md): the statistical protocol
(section 11.3), the scenario suite runner, and metrics/report generation for the full
evaluation report (Week 11, docs/design/13_roadmap.md).
"""

from atlas.evaluation.metrics import (
    QueueSampler,
    TrafficMetrics,
    TripRecord,
    compute_traffic_metrics,
    parse_tripinfo,
)
from atlas.evaluation.runner import run_scenario_cell
from atlas.evaluation.statistics import (
    ComparisonResult,
    ConfidenceInterval,
    cohens_d,
    compare_two_controllers,
    holm_bonferroni,
    mean_confidence_interval,
    welch_t_test,
)

__all__ = [
    "ConfidenceInterval",
    "mean_confidence_interval",
    "welch_t_test",
    "cohens_d",
    "holm_bonferroni",
    "ComparisonResult",
    "compare_two_controllers",
    "TripRecord",
    "TrafficMetrics",
    "QueueSampler",
    "parse_tripinfo",
    "compute_traffic_metrics",
    "run_scenario_cell",
]
