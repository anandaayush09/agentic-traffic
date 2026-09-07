"""Corridor-level coordination (docs/design/04_agent_architecture.md section 4.2):
algorithmic (not learned) green-wave offset computation and spillback detection, plus
(Week 7 part 2+) the Corridor Coordinator service that wires these into agent
observations as advisory hints.
"""

from atlas.coordination.greenwave import compute_greenwave_offsets, detect_spillback

__all__ = ["compute_greenwave_offsets", "detect_spillback"]
