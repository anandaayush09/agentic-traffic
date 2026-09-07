"""The cognitive plane (docs/design/04_agent_architecture.md sections 4.4-4.5): the
playbook registry, the deterministic mapper (the provider chain's terminal, always-
available fallback), the provider chain itself, the what-if tool, and the LangGraph
Supervisor graph tying all of it together.
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
from atlas.cognitive.provider_chain import (
    PlaybookProvider,
    ProviderChain,
    ProviderError,
    ProviderRateLimitedError,
    ProviderUnavailableError,
    TokenBucket,
)
from atlas.cognitive.supervisor import SupervisorState, build_supervisor_graph, initial_state
from atlas.cognitive.what_if import WhatIfResult, run_what_if

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
    "PlaybookProvider",
    "ProviderChain",
    "ProviderError",
    "ProviderRateLimitedError",
    "ProviderUnavailableError",
    "TokenBucket",
    "WhatIfResult",
    "run_what_if",
    "SupervisorState",
    "build_supervisor_graph",
    "initial_state",
]
