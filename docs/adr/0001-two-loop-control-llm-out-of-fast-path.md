# ADR-0001: Two-loop control — LLM stays out of the fast path

**Status:** Accepted (design Part 3; realized by the control/cognitive plane split built
across Weeks 3–9)

## Context

ATLAS needs both sub-second signal control decisions (every 5 simulated seconds, 25
junctions) and reasoned, multi-step incident response (diverting traffic, metering, bus
bridges). A single "agentic" loop that let an LLM choose signal phases directly would be the
simplest architecture to describe, but LLM inference latency (seconds, not milliseconds) and
free-tier rate limits make it structurally incompatible with real-time signal control.

## Decision

Split control into two loops that never share a failure domain:

- **Fast loop** (control plane): MAPPO policy + safety shield + classical fallbacks, entirely
  local computation, no network calls off-box, sub-second decisions.
- **Slow loop** (cognitive plane): LangGraph Supervisor consuming incident/anomaly/forecast
  events, reasoning over playbooks with LLM assistance, producing `playbook.apply` events the
  control plane interprets as mode/parameter changes — never raw phase commands.

## Alternatives considered

- **LLM-in-the-loop control** (LLM chooses phases directly): rejected — no LLM meets a 1s p99
  latency budget under free-tier limits, and it would make the safety-critical path dependent
  on an external API's availability and cost.

## Consequences

- The fast loop's correctness can be verified and chaos-tested (`tests/chaos/`) with zero LLM
  dependency — verified for real in Week 11 with the LLM chain entirely absent (no provider
  keys configured in this environment) and the system still functioning end to end via the
  deterministic mapper (ADR-0010).
- The cognitive plane's slow reasoning never blocks or is blocked by signal control; an
  incident response taking minutes to resolve has zero effect on per-junction decision
  cadence.
- Cost: playbooks (ADR-0011) are the only channel through which the cognitive plane can
  affect control — genuinely novel operator interventions (a plan the playbook set doesn't
  cover) require adding a new playbook, not just a better prompt.
