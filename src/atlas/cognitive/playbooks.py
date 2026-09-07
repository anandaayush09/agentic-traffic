"""The playbook registry: parameterized, pre-modeled response templates (docs/design/
04_agent_architecture.md section 4.4: "Playbooks *are* the plans... The LLM's job is
selection, parameterization, sequencing, and stand-down -- not free-form control. This
mirrors real TMC practice and bounds the blast radius of model error."). Six playbooks,
within the design's stated "5-8 playbooks" range, covering every `incident.events.v1`
class (Week 9's own new schema).

Deliberately NOT free-form: every playbook is a named template with a typed parameter
schema (a pydantic model), not a string an LLM writes from scratch. The Supervisor
(and, until it exists, the deterministic mapper) SELECTS a playbook and fills in its
parameters -- it cannot invent a new kind of action. This is what "bounds the blast
radius of model error" means concretely: whatever the LLM (or its deterministic
stand-in) decides, the SET of possible actions is fixed and small, and every one of
them is itself just a signal-control primitive this project already knows how to apply
safely (mode changes on IntersectionAgentService, a Preemption Overlay-style lock, or
a corridor-hint update) -- never a raw command to the twin.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class DivertArterialParams(BaseModel):
    from_edge: str
    to_edge: str
    duration_s: float = Field(gt=0)


class MeterUpstreamParams(BaseModel):
    corridor_id: str
    rate: float = Field(gt=0, le=1, description="Fraction of normal upstream inflow to admit")


class BusBridgeParams(BaseModel):
    route_id: str


class WeatherCapacityDerateParams(BaseModel):
    area_id: str
    factor: float = Field(gt=0, lt=1, description="Fraction of normal capacity retained")


class SwitchToMaxPressureParams(BaseModel):
    junction_ids: list[str] = Field(min_length=1)


class HoldCurrentPlanParams(BaseModel):
    reason: str


class PlaybookSpec(BaseModel):
    name: str
    description: str
    params_model: type[BaseModel]
    """A playbook is only ever parameterized by its OWN params model -- the
    Supervisor/mapper cannot construct a DivertArterialParams for a bus_bridge
    playbook, by construction, not by convention."""

    model_config = {"arbitrary_types_allowed": True}


class PlaybookProposal(BaseModel):
    """A playbook name + validated parameters -- what the Supervisor (or the
    deterministic mapper) actually proposes for one incident. Mirrors
    `playbook.proposed`'s shape (docs/design/04_agent_architecture.md section 4.4).
    """

    playbook_name: str
    params: dict[str, object]
    rationale: str


class PlaybookRegistry:
    def __init__(self, specs: list[PlaybookSpec] | None = None) -> None:
        self._specs: dict[str, PlaybookSpec] = {
            spec.name: spec for spec in (specs or DEFAULT_PLAYBOOKS)
        }

    def get(self, name: str) -> PlaybookSpec:
        if name not in self._specs:
            raise KeyError(f"No playbook named '{name}' in the registry")
        return self._specs[name]

    def names(self) -> list[str]:
        return sorted(self._specs)

    def validate_proposal(self, proposal: PlaybookProposal) -> BaseModel:
        """Validates `proposal.params` against its playbook's OWN params model --
        raises pydantic's ValidationError if a param is missing/wrong-typed, or KeyError
        if the playbook name doesn't exist at all. Returns the validated params model
        instance, ready to compile into whatever mode/limit changes the control plane
        actually applies (docs/design/04_agent_architecture.md section 4.4: "playbooks
        compile to mode/limit changes... never writes control.commands directly").
        """
        spec = self.get(proposal.playbook_name)
        return spec.params_model(**proposal.params)


DEFAULT_PLAYBOOKS: list[PlaybookSpec] = [
    PlaybookSpec(
        name="divert_arterial",
        description="Reroute through-traffic from one arterial edge to an alternate for a bounded duration.",
        params_model=DivertArterialParams,
    ),
    PlaybookSpec(
        name="meter_upstream",
        description="Restrict inflow into a corridor to a fraction of normal demand, relieving downstream congestion.",
        params_model=MeterUpstreamParams,
    ),
    PlaybookSpec(
        name="bus_bridge",
        description="Prioritize a bus route's signal requests along its corridor.",
        params_model=BusBridgeParams,
    ),
    PlaybookSpec(
        name="weather_capacity_derate",
        description="Reduce assumed road capacity in an area to reflect degraded weather conditions.",
        params_model=WeatherCapacityDerateParams,
    ),
    PlaybookSpec(
        name="switch_to_max_pressure",
        description="Fall back specific junctions to the deterministic max-pressure controller -- the safe default when sensor/comms telemetry for those junctions is degraded.",
        params_model=SwitchToMaxPressureParams,
    ),
    PlaybookSpec(
        name="hold_current_plan",
        description="Take no control action; keep monitoring. The safe default when there isn't yet enough evidence to justify a disruptive change.",
        params_model=HoldCurrentPlanParams,
    ),
]
