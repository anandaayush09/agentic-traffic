# ADR-0011: Playbooks as bounded, pre-modeled response templates

**Status:** Accepted (design Part 4 §4.4; built Week 9 part 3, commit `bee4719`)

## Context

If the Supervisor's LLM were allowed to emit free-form control actions or arbitrary
parameters, a single bad completion could apply an unbounded or nonsensical change to live
traffic control. The cognitive plane needs a way to let the LLM contribute judgment
(incident triage, response selection, parameterization) without letting it control the blast
radius of its own mistakes.

## Decision

A fixed registry of playbooks (`divert_arterial`, `meter_upstream`, `bus_bridge`,
`weather_capacity_derate`, `switch_to_max_pressure`, `hold_current_plan`), each with its own
typed Pydantic parameters model. The LLM's role is selection, parameterization, sequencing,
and stand-down — never inventing a new class of action. This mirrors real Traffic Management
Center practice, where operators work from a known playbook set rather than improvising novel
interventions live.

## Alternatives considered

- **Free-form LLM control** (the model generates an arbitrary structured or unstructured
  action): rejected — this is the same "ungrounded, unbounded agentic action" failure mode
  ADR-0001 and ADR-0006 both guard against, applied to the parameterization step
  specifically.

## Consequences

- Verified directly (Week 9 part 3): one playbook's typed params model cannot pass as
  another's — the concrete, testable mechanism behind "bounds the blast radius of model
  error," not just an assertion.
- A genuinely novel operator intervention that no existing playbook covers requires adding a
  new playbook (a deliberate, reviewed code change) rather than a better prompt — a real
  limitation of this design, traded deliberately for predictability and auditability.
- The deterministic mapper (ADR-0010) and the LLM path share the exact same playbook
  registry and parameter types — "scripted" and "reasoned" modes differ only in *how* a
  playbook is selected and parameterized, never in what a playbook is allowed to do.
