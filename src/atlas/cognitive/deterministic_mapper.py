"""The deterministic mapper: incident class -> default playbook, no LLM involved
(docs/design/04_agent_architecture.md section 4.4: "if the LLM chain is exhausted/down,
the deterministic mapper (incident class -> default playbook) keeps responses
flowing"). This is the terminal link in the provider chain (Groq -> Gemini -> Ollama ->
deterministic mapper) -- when every LLM option is unavailable, THIS still produces a
valid, schema-conformant playbook proposal, just without any LLM-specific reasoning
about the incident's particulars.

The mapping is deliberately conservative: every class maps to either a narrowly-scoped
mitigation (divert/meter/derate) or, where a disruptive action without more context
would itself be risky (an emergency/VIP run the Preemption Overlay already handles
separately, or a genuinely unknown incident class), `hold_current_plan` -- "do nothing
yet, keep monitoring" is always a safe, valid deterministic default; guessing at a
disruptive playbook parameterization with no LLM reasoning behind it would not be.
"""

from __future__ import annotations

from collections.abc import Callable

from atlas.cognitive.playbooks import PlaybookProposal

IncidentClass = str  # matches schemas/incident.events.v1.schema.json's "class" enum
_MapperFn = Callable[[dict[str, object]], PlaybookProposal]

# One default-parameterization function per incident class -- kept as small closures
# rather than a static params dict, since even the "no LLM" fallback needs the
# incident's own location/link_ids to fill in a playbook's required parameters.
_DEFAULT_PLAYBOOK_FOR_CLASS: dict[str, _MapperFn] = {}


def _register(incident_class: str) -> Callable[[_MapperFn], _MapperFn]:
    def decorator(fn: _MapperFn) -> _MapperFn:
        _DEFAULT_PLAYBOOK_FOR_CLASS[incident_class] = fn
        return fn  # returns the original function unchanged, so e.g. _map_comms_fault
        # can still call _map_sensor_fault(...) directly elsewhere in this module --
        # a decorator that returned None here would clobber that name's callability.

    return decorator


def _first_link_id(incident: dict[str, object]) -> str:
    location = incident.get("location")
    if isinstance(location, dict):
        link_ids = location.get("link_ids")
        if isinstance(link_ids, list) and link_ids:
            return str(link_ids[0])
    return "unknown_link"


@_register("accident")
def _map_accident(incident: dict[str, object]) -> PlaybookProposal:
    edge = _first_link_id(incident)
    return PlaybookProposal(
        playbook_name="divert_arterial",
        params={"from_edge": edge, "to_edge": edge, "duration_s": 600.0},
        rationale=(
            f"Deterministic default for class='accident': divert around '{edge}' for "
            "10 minutes -- no LLM available to pick a real alternate edge or a "
            "duration informed by what-if evidence."
        ),
    )


@_register("closure")
def _map_closure(incident: dict[str, object]) -> PlaybookProposal:
    edge = _first_link_id(incident)
    return PlaybookProposal(
        playbook_name="divert_arterial",
        params={"from_edge": edge, "to_edge": edge, "duration_s": 1800.0},
        rationale=f"Deterministic default for class='closure': divert around '{edge}' for 30 minutes.",
    )


@_register("construction")
def _map_construction(incident: dict[str, object]) -> PlaybookProposal:
    edge = _first_link_id(incident)
    return PlaybookProposal(
        playbook_name="meter_upstream",
        params={"corridor_id": edge, "rate": 0.7},
        rationale=f"Deterministic default for class='construction': meter inflow into '{edge}' to 70%.",
    )


@_register("weather")
def _map_weather(incident: dict[str, object]) -> PlaybookProposal:
    edge = _first_link_id(incident)
    return PlaybookProposal(
        playbook_name="weather_capacity_derate",
        params={"area_id": edge, "factor": 0.6},
        rationale="Deterministic default for class='weather': derate assumed capacity to 60%.",
    )


@_register("event_surge")
def _map_event_surge(incident: dict[str, object]) -> PlaybookProposal:
    edge = _first_link_id(incident)
    return PlaybookProposal(
        playbook_name="meter_upstream",
        params={"corridor_id": edge, "rate": 0.8},
        rationale=f"Deterministic default for class='event_surge': meter inflow into '{edge}' to 80%.",
    )


@_register("sensor_fault")
def _map_sensor_fault(incident: dict[str, object]) -> PlaybookProposal:
    location = incident.get("location")
    junction_ids = (
        location.get("link_ids", ["unknown_junction"])
        if isinstance(location, dict)
        else ["unknown_junction"]
    )
    return PlaybookProposal(
        playbook_name="switch_to_max_pressure",
        params={"junction_ids": junction_ids},
        rationale="Deterministic default for class='sensor_fault': fall back to max-pressure at the affected junction(s), the safe response to degraded telemetry.",
    )


@_register("comms_fault")
def _map_comms_fault(incident: dict[str, object]) -> PlaybookProposal:
    return _map_sensor_fault(incident)


@_register("emergency_run")
def _map_emergency_run(incident: dict[str, object]) -> PlaybookProposal:
    return PlaybookProposal(
        playbook_name="hold_current_plan",
        params={
            "reason": "emergency_run is handled by the Preemption Overlay directly, not a playbook."
        },
        rationale="Deterministic default for class='emergency_run': no playbook action -- preemption is a separate, deterministic mechanism (docs/design/04_agent_architecture.md section 4.3).",
    )


@_register("vip_run")
def _map_vip_run(incident: dict[str, object]) -> PlaybookProposal:
    return _map_emergency_run(incident)


@_register("unknown")
def _map_unknown(incident: dict[str, object]) -> PlaybookProposal:
    return PlaybookProposal(
        playbook_name="hold_current_plan",
        params={
            "reason": "Incident class is 'unknown' -- no safe deterministic action without more evidence."
        },
        rationale="Deterministic default for class='unknown': hold and keep monitoring rather than guess.",
    )


def map_incident_to_playbook(incident: dict[str, object]) -> PlaybookProposal:
    """`incident`: a payload matching schemas/incident.events.v1.schema.json. Raises
    KeyError if `incident["class"]` isn't a value that schema allows -- a schema-valid
    incident should never reach this function with an unmapped class, since every
    enum value has a registered default above.
    """
    incident_class = incident["class"]
    if incident_class not in _DEFAULT_PLAYBOOK_FOR_CLASS:
        raise KeyError(f"No deterministic mapping registered for incident class '{incident_class}'")
    return _DEFAULT_PLAYBOOK_FOR_CLASS[incident_class](incident)
