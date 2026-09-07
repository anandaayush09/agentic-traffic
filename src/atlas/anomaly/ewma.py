"""EWMA/z-score anomaly detection -- tier 1 of the two-tier detector (docs/design/
04_agent_architecture.md section 4.8: "EWMA/z-score per lane for fast, explainable
deviations (sudden occupancy spikes, flatlined sensors -- sensor-fault signatures)").
Zero training, always available -- the tier that "keeps working when the ML tier is
down, same degradation philosophy as control" (that section's own words).

Two distinct checks, not one, because they're different failure signatures with
different statistical shapes:
- **Spike**: a z-score test against a running EWMA mean/variance -- catches sudden
  jumps (an accident, a sensor glitch reporting a huge value).
- **Flatline**: a sensor whose value stops changing at all -- a z-score test alone
  CANNOT catch this (a value that exactly matches its own recent history has z~0 by
  construction, no matter how implausible "the exact same reading N times in a row"
  is for a live traffic sensor) -- verified directly in tests/unit/test_ewma.py rather
  than assumed, since this is the kind of gap that looks fine until a real flatlined
  sensor sails through undetected.

EWMA mean/variance update is the incremental (Finch 2009 / Welford-style) formula, using
the variance estimate from BEFORE this observation to compute its z-score -- checking a
point against history that doesn't already include it, not against a mean/variance the
point itself just pulled toward.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal

AnomalyKind = Literal["spike", "flatline"]


@dataclass(frozen=True, slots=True)
class AnomalyEvent:
    key: str
    kind: AnomalyKind
    value: float
    z_score: float | None
    """None for a flatline event -- z-score is undefined/meaningless there (variance
    is ~0 by definition of "flatlined"), not omitted by oversight."""


@dataclass
class _StreamState:
    mean: float
    var: float
    recent: deque[float]
    initialized: bool = False


class EwmaAnomalyDetector:
    def __init__(
        self,
        alpha: float = 0.3,
        z_threshold: float = 3.0,
        flatline_window: int = 10,
        flatline_eps: float = 1e-9,
        var_floor: float = 1e-6,
    ) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self._alpha = alpha
        self._z_threshold = z_threshold
        self._flatline_window = flatline_window
        self._flatline_eps = flatline_eps
        self._var_floor = var_floor
        self._streams: dict[str, _StreamState] = {}

    def update(self, key: str, value: float) -> AnomalyEvent | None:
        state = self._streams.get(key)
        if state is None:
            state = _StreamState(
                mean=value, var=0.0, recent=deque(maxlen=self._flatline_window), initialized=True
            )
            self._streams[key] = state
            state.recent.append(value)
            return None  # first observation: nothing to compare against yet

        prior_mean, prior_var = state.mean, state.var
        diff = value - prior_mean
        increment = self._alpha * diff
        state.mean = prior_mean + increment
        state.var = (1 - self._alpha) * (prior_var + diff * increment)
        state.recent.append(value)

        if self._is_flatlined(state):
            return AnomalyEvent(key=key, kind="flatline", value=value, z_score=None)

        z_score = diff / (prior_var**0.5 + self._var_floor)
        if abs(z_score) >= self._z_threshold:
            return AnomalyEvent(key=key, kind="spike", value=value, z_score=z_score)

        return None

    def current_mean_var(self, key: str) -> tuple[float, float]:
        """The running EWMA mean/variance for `key` -- exposed for verification (see
        tests/unit/test_ewma.py's hand-traced recursion) and for a caller that wants to
        report the detector's own internal state, not just its anomaly events.
        """
        state = self._streams[key]
        return state.mean, state.var

    def _is_flatlined(self, state: _StreamState) -> bool:
        if len(state.recent) < self._flatline_window:
            return False
        first = state.recent[0]
        return all(abs(v - first) <= self._flatline_eps for v in state.recent)
