"""Traffic metrics computation (docs/design/11_evaluation.md section 11.2) from a
real completed SUMO run's own tripinfo output -- not derived indirectly from other
signals. SUMO's `--tripinfo-output` (verified with a real smoke run before writing
this: `<tripinfo id=... duration=... waitingTime=... arrival=.../>` per vehicle,
one element per trip) is the standard, most reliable source for per-vehicle
waiting/travel-time metrics; queue metrics need separate per-step sampling since
tripinfo only reports per-vehicle, not per-lane, state.

Scope, deliberately: implements "Mean waiting time," "Mean travel time / delay," and
"Throughput," plus queue mean/p95 (docs/design/11_evaluation.md section 11.2's first
four rows). NOT implemented: emergency response/recovery time (Week 7 already
measures this separately, per-scenario, in scripts/eval_emergency_preemption.py, not
a per-run aggregate metric this module produces), fuel/CO2 (needs SUMO's HBEFA
emission output, a separate `--emission-output` flag not wired up), PT efficiency and
person-delay (need per-vehicle-type breakdowns this module doesn't compute yet). Real,
named gaps -- not silently claimed covered.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree


@dataclass(frozen=True, slots=True)
class TripRecord:
    vehicle_id: str
    duration_s: float
    waiting_time_s: float
    completed: bool
    """False for a trip SUMO wrote via --tripinfo-output.write-unfinished because the
    vehicle was still en route when the episode ended (no `arrival` timestamp) --
    still real data (the vehicle really did wait that long), just excluded from the
    throughput count, which by definition only counts finished trips."""


def parse_tripinfo(path: Path) -> list[TripRecord]:
    root = ElementTree.parse(path).getroot()
    records = []
    for elem in root.findall("tripinfo"):
        arrival = elem.get("arrival", "-1")
        records.append(
            TripRecord(
                vehicle_id=elem.get("id", ""),
                duration_s=float(elem.get("duration", "0")),
                waiting_time_s=float(elem.get("waitingTime", "0")),
                completed=float(arrival) >= 0,
            )
        )
    return records


def _percentile(values: list[float], p: float) -> float:
    """Linear-interpolation percentile (numpy's default 'linear' method) -- hand-
    verified in tests against small hand-sorted lists, not trusted from a library
    without checking, since queue robustness claims rest on this being right.
    """
    if not values:
        raise ValueError("Cannot compute a percentile of an empty list")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = p * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


@dataclass(frozen=True, slots=True)
class TrafficMetrics:
    mean_waiting_time_s: float
    mean_travel_time_s: float
    throughput_veh_per_hour: float
    mean_queue_length: float
    p95_queue_length: float
    num_trips: int
    num_completed_trips: int


def compute_traffic_metrics(
    trips: list[TripRecord], queue_samples: list[float], episode_duration_s: float
) -> TrafficMetrics:
    """`queue_samples`: every per-step, per-lane halting-vehicle-count observation
    taken during the episode (see `QueueSampler` below) -- NOT one sample per lane,
    one sample per (step, lane) pair, so `p95_queue_length` reflects how bad queues
    got at their worst moments, not just on average.
    """
    if not trips:
        raise ValueError("Cannot compute traffic metrics from zero trips")
    if not queue_samples:
        raise ValueError("Cannot compute queue metrics from zero samples")
    if episode_duration_s <= 0:
        raise ValueError("episode_duration_s must be positive")

    completed = [t for t in trips if t.completed]
    return TrafficMetrics(
        mean_waiting_time_s=sum(t.waiting_time_s for t in trips) / len(trips),
        mean_travel_time_s=sum(t.duration_s for t in trips) / len(trips),
        throughput_veh_per_hour=len(completed) / (episode_duration_s / 3600.0),
        mean_queue_length=sum(queue_samples) / len(queue_samples),
        p95_queue_length=_percentile(queue_samples, 0.95),
        num_trips=len(trips),
        num_completed_trips=len(completed),
    )


class QueueSampler:
    """Accumulates per-step, per-lane halting-vehicle counts over the course of an
    episode -- a thin wrapper so a runner's step loop can call `sample()` once per
    decision interval without reimplementing the bookkeeping each time.
    """

    def __init__(self) -> None:
        self._samples: list[float] = []

    def sample(self, conn: object, lane_ids: list[str]) -> None:
        for lane_id in lane_ids:
            self._samples.append(float(conn.lane.getLastStepHaltingNumber(lane_id)))  # type: ignore[attr-defined]

    @property
    def samples(self) -> list[float]:
        return list(self._samples)
