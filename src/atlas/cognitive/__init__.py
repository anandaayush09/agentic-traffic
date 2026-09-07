"""The cognitive plane (docs/design/04_agent_architecture.md sections 4.4-4.5): the
playbook registry, the deterministic mapper (the provider chain's terminal, always-
available fallback), and -- later this week -- the provider chain and the LangGraph
Supervisor graph itself.
"""

from atlas.cognitive.deterministic_mapper import map_incident_to_playbook
from atlas.cognitive.playbooks import (
    DEFAULT_PLAYBOOKS,
    BusBridgeParams,
    DivertArterialParams,
    HoldCurrentPlanParams,
    MeterUpstreamParams,
    PlaybookProposal,
    PlaybookRegistry,
    PlaybookSpec,
    SwitchToMaxPressureParams,
    WeatherCapacityDerateParams,
)

__all__ = [
    "PlaybookSpec",
    "PlaybookProposal",
    "PlaybookRegistry",
    "DEFAULT_PLAYBOOKS",
    "DivertArterialParams",
    "MeterUpstreamParams",
    "BusBridgeParams",
    "WeatherCapacityDerateParams",
    "SwitchToMaxPressureParams",
    "HoldCurrentPlanParams",
    "map_incident_to_playbook",
]
