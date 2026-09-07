"""FastAPI/async control-plane services (docs/design/09_repository_design.md's
`services/`) -- Week 6's first real services: the Intersection Agent (one junction's
decision loop) and the Watchdog (heartbeat staleness detection / fallback trigger).
"""

from atlas.services.corridor_coordinator import (
    CorridorCoordinatorConfig,
    CorridorCoordinatorService,
)
from atlas.services.decision import AgentMode, actor_name_for_mode, select_action_for_mode
from atlas.services.intersection_agent import IntersectionAgentConfig, IntersectionAgentService
from atlas.services.watchdog import Watchdog

__all__ = [
    "AgentMode",
    "actor_name_for_mode",
    "select_action_for_mode",
    "IntersectionAgentConfig",
    "IntersectionAgentService",
    "CorridorCoordinatorConfig",
    "CorridorCoordinatorService",
    "Watchdog",
]
