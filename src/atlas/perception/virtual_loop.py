"""Virtual-loop counting: line-crossing on tracked boxes (docs/design/06_ai_pipeline.md
section 6.3, deliverable 2: "virtual-loop counting (line-crossing on tracked boxes --
ByteTrack) so cameras emit the same count/occupancy semantics as loops"). This is the
counting geometry itself, deliberately independent of whatever produced the tracked
positions -- a real ByteTrack tracker feeding centroids frame-by-frame, or (as in this
project's tests, since no GPU/YOLO/UA-DETRAC pipeline runs locally -- see the ONNX/YOLO
scripts' own docstrings) a synthetic sequence of positions. `LineCrossingCounter` only
needs `update()` called once per track per frame; it has no notion of "camera" or
"detector" at all.

Occupancy is a coarser signal than count: the fraction of updates where at least one
track was within `proximity_m` of the loop line, over the caller's own window -- a
simplification (real loop-detector occupancy is "fraction of time covered", not "fraction
of samples near"), but a fair semantic to expose alongside count for this project's
purposes: comparing camera-derived counts/occupancy to a SUMO induction loop's own
`getLastStepOccupancy`-style metric (atlas.twin.env.LaneState.occupancy), not claiming
metrological equivalence to a physical loop detector.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class VirtualLoop:
    """A line segment (p1 -> p2) in the camera's ground-plane coordinates -- the same
    role a physical induction loop plays, but drawn in software instead of cut into
    the road.
    """

    p1: tuple[float, float]
    p2: tuple[float, float]


def _signed_side(loop: VirtualLoop, point: tuple[float, float]) -> float:
    """Positive/negative/zero depending on which side of the loop's line `point` is
    on -- the standard 2D cross-product line-side test. A crossing is exactly a sign
    change between two consecutive calls for the same track.
    """
    (x1, y1), (x2, y2) = loop.p1, loop.p2
    x, y = point
    return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)


@dataclass
class LineCrossingCounter:
    loop: VirtualLoop
    count: int = field(default=0, init=False)
    _last_side: dict[str, float] = field(default_factory=dict, init=False)

    def update(self, track_id: str, position: tuple[float, float]) -> bool:
        """Feed one frame's position for one track. Returns True exactly when this
        update registered a new crossing (the track's side of the loop's line flipped
        sign since its last update) -- a track sitting exactly ON the line, or being
        seen for the first time, never counts as a crossing on its own.
        """
        side = _signed_side(self.loop, position)
        previous = self._last_side.get(track_id)
        crossed = (
            previous is not None and previous != 0 and side != 0 and (previous > 0) != (side > 0)
        )
        if crossed:
            self.count += 1
        self._last_side[track_id] = side
        return crossed

    def reset_count(self) -> None:
        """Zeros the running count without forgetting each track's current side --
        the per-message-interval reset a `vision.detections.v1` producer needs (report
        counts since the last publish, not since the tracker started), while still
        correctly detecting a crossing that straddles the reset boundary.
        """
        self.count = 0


def occupancy_fraction(
    loop: VirtualLoop, positions: list[tuple[float, float]], proximity_m: float
) -> float:
    """The fraction of `positions` within `proximity_m` of the loop's line segment --
    one frame's worth of "is anything on the loop right now" samples, meant to be
    averaged by the caller over a reporting window (see the module docstring's
    occupancy-semantics caveat).
    """
    if not positions:
        return 0.0
    near = sum(1 for p in positions if _distance_to_segment(loop, p) <= proximity_m)
    return near / len(positions)


def _distance_to_segment(loop: VirtualLoop, point: tuple[float, float]) -> float:
    (x1, y1), (x2, y2) = loop.p1, loop.p2
    x, y = point
    dx, dy = x2 - x1, y2 - y1
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return float(((x - x1) ** 2 + (y - y1) ** 2) ** 0.5)
    t = max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / length_sq))
    proj_x, proj_y = x1 + t * dx, y1 + t * dy
    return float(((x - proj_x) ** 2 + (y - proj_y) ** 2) ** 0.5)
