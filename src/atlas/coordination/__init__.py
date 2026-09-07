"""Corridor-level coordination and emergency preemption (docs/design/
04_agent_architecture.md sections 4.2-4.3): algorithmic (not learned) green-wave offset
computation, spillback detection, and the deterministic Preemption Overlay state
machine + lock protocol. The live FastAPI services that wire these to a running twin
and the bus live in `atlas.services`.
"""

from atlas.coordination.corridor_geometry import compute_corridor_segment_distances_m
from atlas.coordination.greenwave import compute_greenwave_offsets, detect_spillback
from atlas.coordination.lock_store import InMemoryLockStore, LockStore
from atlas.coordination.preemption import EmergencyRun, PreemptionOverlay, RouteStop

__all__ = [
    "compute_greenwave_offsets",
    "detect_spillback",
    "compute_corridor_segment_distances_m",
    "LockStore",
    "InMemoryLockStore",
    "PreemptionOverlay",
    "EmergencyRun",
    "RouteStop",
]
