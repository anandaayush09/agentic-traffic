"""The scripted incident suite (docs/design/13_roadmap.md's Week 9 test bullet:
"scripted incident suite v1 (10 incidents) scoring playbook accuracy + grounding
rate"). Ten incidents, one per `incident.events.v1` class, each hand-labeled with the
playbook a human TMC operator would pick for it (`expected_playbook`, the accuracy
ground truth) and built from REAL manhattan_midtown edge/junction ids already used
elsewhere in this project's evaluation scripts and tests -- so a proposal's own
location-based params can be checked against the incident's own evidence, not a
template string.

Run through a `ProviderChain` with no real LLM providers configured (no API keys exist
in this environment, same stance as `provider_chain.py`), every incident here exercises
the terminal deterministic-mapper fallback for real -- this IS docs/design/12_risks.md
R4's "the demo runs on deterministic playbooks... system value survives by design"
made literal across the whole suite, not just the isolated fake-provider unit tests in
`test_provider_chain.py`. A real LLM provider slotting in later changes `provider_used`
and, per docs/design/12_risks.md's own framing, is expected to matter most on grounding
rate for the incidents the deterministic mapper can't parameterize freely (accident/
closure -- it always picks the incident's OWN edge as both `from_edge` and `to_edge`,
since it has no real alternate-route reasoning; an LLM with map context could do
better).
"""

from __future__ import annotations

from dataclasses import dataclass

from atlas.cognitive.playbooks import PlaybookProposal, PlaybookRegistry
from atlas.cognitive.provider_chain import ProviderChain

# Real ids reused from scripts/eval_emergency_preemption.py, scripts/eval_anomaly_detection.py,
# and scripts/train_mappo_corridor.py -- not fabricated for this suite.
_CORRIDOR_JUNCTIONS = ["42430333", "42430338", "42430342", "42430344", "42430347"]
_ACCIDENT_EDGE = "448686048#0"
_UNAFFECTED_EDGE = "127693103#0"
_ROUTE_EDGE_B = "464516471#0"


@dataclass(frozen=True, slots=True)
class ScriptedIncident:
    incident_id: str
    incident: dict[str, object]
    expected_playbook: str
    """The playbook a human TMC operator would pick for this incident -- the accuracy
    ground truth. Not necessarily what any mapper (deterministic or LLM) will always
    produce; scoring against this is exactly the point."""


SCRIPTED_INCIDENTS: list[ScriptedIncident] = [
    ScriptedIncident(
        incident_id="inc-accident-01",
        incident={
            "msg_id": "00000000-0000-0000-0000-000000000001",
            "ts": "2026-01-01T08:00:00Z",
            "schema_version": "1",
            "incident_id": "inc-accident-01",
            "class": "accident",
            "status": "detected",
            "location": {"link_ids": [_ACCIDENT_EDGE]},
            "confidence": 0.9,
        },
        expected_playbook="divert_arterial",
    ),
    ScriptedIncident(
        incident_id="inc-closure-01",
        incident={
            "msg_id": "00000000-0000-0000-0000-000000000002",
            "ts": "2026-01-01T08:05:00Z",
            "schema_version": "1",
            "incident_id": "inc-closure-01",
            "class": "closure",
            "status": "detected",
            "location": {"link_ids": [_ROUTE_EDGE_B]},
            "confidence": 0.95,
        },
        expected_playbook="divert_arterial",
    ),
    ScriptedIncident(
        incident_id="inc-construction-01",
        incident={
            "msg_id": "00000000-0000-0000-0000-000000000003",
            "ts": "2026-01-01T08:10:00Z",
            "schema_version": "1",
            "incident_id": "inc-construction-01",
            "class": "construction",
            "status": "detected",
            "location": {"link_ids": [_UNAFFECTED_EDGE]},
            "confidence": 1.0,
        },
        expected_playbook="meter_upstream",
    ),
    ScriptedIncident(
        incident_id="inc-weather-01",
        incident={
            "msg_id": "00000000-0000-0000-0000-000000000004",
            "ts": "2026-01-01T08:15:00Z",
            "schema_version": "1",
            "incident_id": "inc-weather-01",
            "class": "weather",
            "status": "detected",
            "location": {"link_ids": [_ACCIDENT_EDGE]},
            "confidence": 0.7,
        },
        expected_playbook="weather_capacity_derate",
    ),
    ScriptedIncident(
        incident_id="inc-event-surge-01",
        incident={
            "msg_id": "00000000-0000-0000-0000-000000000005",
            "ts": "2026-01-01T08:20:00Z",
            "schema_version": "1",
            "incident_id": "inc-event-surge-01",
            "class": "event_surge",
            "status": "detected",
            "location": {"link_ids": [_UNAFFECTED_EDGE]},
            "confidence": 0.8,
        },
        expected_playbook="meter_upstream",
    ),
    ScriptedIncident(
        incident_id="inc-sensor-fault-01",
        incident={
            "msg_id": "00000000-0000-0000-0000-000000000006",
            "ts": "2026-01-01T08:25:00Z",
            "schema_version": "1",
            "incident_id": "inc-sensor-fault-01",
            "class": "sensor_fault",
            "status": "detected",
            "location": {"link_ids": [_CORRIDOR_JUNCTIONS[0]]},
            "confidence": 0.85,
        },
        expected_playbook="switch_to_max_pressure",
    ),
    ScriptedIncident(
        incident_id="inc-comms-fault-01",
        incident={
            "msg_id": "00000000-0000-0000-0000-000000000007",
            "ts": "2026-01-01T08:30:00Z",
            "schema_version": "1",
            "incident_id": "inc-comms-fault-01",
            "class": "comms_fault",
            "status": "detected",
            "location": {"link_ids": _CORRIDOR_JUNCTIONS[1:3]},
            "confidence": 0.6,
        },
        expected_playbook="switch_to_max_pressure",
    ),
    ScriptedIncident(
        incident_id="inc-emergency-run-01",
        incident={
            "msg_id": "00000000-0000-0000-0000-000000000008",
            "ts": "2026-01-01T08:35:00Z",
            "schema_version": "1",
            "incident_id": "inc-emergency-run-01",
            "class": "emergency_run",
            "status": "detected",
            "location": {"link_ids": [_ROUTE_EDGE_B]},
            "confidence": 1.0,
        },
        expected_playbook="hold_current_plan",
    ),
    ScriptedIncident(
        incident_id="inc-vip-run-01",
        incident={
            "msg_id": "00000000-0000-0000-0000-000000000009",
            "ts": "2026-01-01T08:40:00Z",
            "schema_version": "1",
            "incident_id": "inc-vip-run-01",
            "class": "vip_run",
            "status": "detected",
            "location": {"link_ids": [_ROUTE_EDGE_B]},
            "confidence": 1.0,
        },
        expected_playbook="hold_current_plan",
    ),
    ScriptedIncident(
        incident_id="inc-unknown-01",
        incident={
            "msg_id": "00000000-0000-0000-0000-000000000010",
            "ts": "2026-01-01T08:45:00Z",
            "schema_version": "1",
            "incident_id": "inc-unknown-01",
            "class": "unknown",
            "status": "detected",
            "confidence": 0.3,
        },
        expected_playbook="hold_current_plan",
    ),
]


def _location_link_ids(incident: dict[str, object]) -> set[str]:
    location = incident.get("location")
    if isinstance(location, dict):
        link_ids = location.get("link_ids")
        if isinstance(link_ids, list):
            return {str(link_id) for link_id in link_ids}
    return set()


def _location_claims_in_params(proposal: PlaybookProposal) -> set[str]:
    """Every string/list-of-strings value in `proposal.params` that plausibly claims a
    real network location (edge or junction id) -- the values a real LLM (or the
    deterministic mapper) could hallucinate if it invented a location it wasn't given.
    `hold_current_plan`'s free-text `reason` is deliberately excluded: it's prose, not a
    location claim.
    """
    claims: set[str] = set()
    for key, value in proposal.params.items():
        if key == "reason":
            continue
        if isinstance(value, str):
            claims.add(value)
        elif isinstance(value, list):
            claims.update(str(item) for item in value if isinstance(item, str))
    return claims


def is_grounded(scripted: ScriptedIncident, proposal: PlaybookProposal) -> bool:
    """A proposal is "grounded" (docs/design/06_ai_pipeline.md section 6.6: "grounding
    rate (% of claims backed by what-if ids)" for the full LLM layer; here, for the
    location claims the deterministic-mapper terminal can make on its own) if every
    location-shaped param value it states is one of the incident's OWN logged
    `location.link_ids` -- i.e. it never invents a location it wasn't given. An
    incident with no location at all (e.g. `unknown`) is trivially grounded by any
    proposal that makes no location claim.
    """
    location_claims = _location_claims_in_params(proposal)
    if not location_claims:
        return True
    return location_claims.issubset(_location_link_ids(scripted.incident))


@dataclass(frozen=True, slots=True)
class IncidentScore:
    incident_id: str
    expected_playbook: str
    actual_playbook: str
    provider_used: str
    playbook_correct: bool
    grounded: bool
    params_valid: bool


@dataclass(frozen=True, slots=True)
class SuiteSummary:
    scores: list[IncidentScore]

    @property
    def playbook_accuracy(self) -> float:
        return sum(s.playbook_correct for s in self.scores) / len(self.scores)

    @property
    def grounding_rate(self) -> float:
        return sum(s.grounded for s in self.scores) / len(self.scores)

    @property
    def param_sanity_rate(self) -> float:
        return sum(s.params_valid for s in self.scores) / len(self.scores)


def score_incident(
    scripted: ScriptedIncident,
    proposal: PlaybookProposal,
    provider_used: str,
    registry: PlaybookRegistry,
) -> IncidentScore:
    try:
        registry.validate_proposal(proposal)
        params_valid = True
    except Exception:  # noqa: BLE001 -- any validation failure (KeyError, pydantic
        # ValidationError) counts as "not sane"; the score cares that it failed, not why.
        params_valid = False

    return IncidentScore(
        incident_id=scripted.incident_id,
        expected_playbook=scripted.expected_playbook,
        actual_playbook=proposal.playbook_name,
        provider_used=provider_used,
        playbook_correct=proposal.playbook_name == scripted.expected_playbook,
        grounded=is_grounded(scripted, proposal),
        params_valid=params_valid,
    )


def run_suite(chain: ProviderChain, registry: PlaybookRegistry) -> SuiteSummary:
    scores: list[IncidentScore] = []
    for scripted in SCRIPTED_INCIDENTS:
        proposal, provider_used = chain.propose(scripted.incident, registry.names())
        scores.append(score_incident(scripted, proposal, provider_used, registry))
    return SuiteSummary(scores=scores)
