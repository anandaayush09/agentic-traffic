"""The sensor fault model (docs/design/03_system_architecture.md section 3.4: "A
sensor fault model (configurable dropout, bias, noise, freeze) sits between SUMO and
the bus"; docs/design/11_evaluation.md section 11.5's detector-dropout robustness
sweep: "0-50% of detectors dead; success criterion: never worse than actuated").

This is the integration `atlas.perception.error_model`'s own module docstring named
as future work ("wiring the resulting per-condition rates into the twin's own
sensor-noise model... is a separate, later integration, not assumed done by this
module") -- built here. `SensorFaultModel.from_error_model_stats` is the real handoff
point: given a real `ErrorModelStats` (Week 8, fitted from actual detection tallies),
its `miss_rate` becomes this model's `dropout_rate` directly. No fresh detection
tallies were fit in this session, so every robustness-sweep script below still
parameterizes `SensorFaultModel` directly with an explicit dropout rate -- honest
about not having exercised the ErrorModelStats handoff with real Week-8-fitted
numbers yet, not silently assumed wired end to end.

Scope: only "dropout" (a detector reporting nothing -- the queue-length reading for
that lane is simply missing) is implemented and wired into a robustness sweep here.
"Freeze" (a detector stuck repeating its last reading) and "bias"/"noise" (a detector
reporting a systematically wrong count) are real, named, NOT-yet-built modes -- adding
them is extending `SensorFaultModel.apply` with the same dict-in/dict-out shape, not a
redesign.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SensorFaultModel:
    dropout_rate: float
    """Fraction of lanes whose queue-length reading is dropped entirely each time
    `apply` is called -- independently per lane, per call (a detector that's down for
    one reading isn't necessarily down for the next; a persistent multi-step outage
    is the caller's own concern to model by re-using the same dropped lane set, not
    this class's)."""
    rng: random.Random

    def __post_init__(self) -> None:
        if not 0.0 <= self.dropout_rate <= 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1], got {self.dropout_rate}")

    def apply(self, queue_lengths: Mapping[str, int]) -> dict[str, int]:
        """Returns a copy of `queue_lengths` with each lane independently dropped
        (removed from the dict entirely) with probability `dropout_rate` -- matching
        `phase_pressure`/`select_max_pressure_phase`'s own existing convention that a
        lane MISSING from `queue_lengths` is treated as empty (0), not present-with-
        value-0 (which would look like "no cars waiting," a materially different and
        wrong claim from "we don't know")."""
        return {
            lane_id: count
            for lane_id, count in queue_lengths.items()
            if self.rng.random() >= self.dropout_rate
        }

    @classmethod
    def from_error_model_stats(cls, miss_rate: float, rng: random.Random) -> SensorFaultModel:
        """The real handoff point named in this module's own docstring: a detector's
        measured miss rate (`atlas.perception.error_model.ErrorModelStats.miss_rate`,
        fitted from real detection tallies) becomes this model's dropout rate
        directly -- a missed detection and a dropped-out reading are the same
        failure mode from the control loop's point of view (a vehicle that should
        have been counted, wasn't).
        """
        return cls(dropout_rate=miss_rate, rng=rng)
