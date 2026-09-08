# ADR-0006: The twin as a tool call, not free-form LLM reasoning

**Status:** Accepted (design Part 3; built Week 9 part 5, commit `ca33ed3`)

## Context

An LLM-based incident commander can *claim* that diverting traffic to arterial B will reduce
congestion, but an unverified claim is just plausible-sounding text — exactly the failure
mode ("agentic AI" as a marketing claim rather than a measured capability) this project set
out to avoid.

## Decision

The digital twin exposes a **what-if** mode as a tool the Supervisor graph can call: fork
current state, apply a hypothetical change, fast-forward 15–30 simulated minutes, return
metric deltas. The LangGraph state machine enforces tool-grounding *by construction*: a
proposal node cannot be reached without first passing through a verification node that calls
what-if, so every claim about a playbook's consequences must cite a real what-if run id.

## Alternatives considered

- **LLM free-form reasoning** (the model asserts an effect from its own world knowledge, no
  simulation check): rejected — this is exactly the ungrounded-claim failure mode the
  design's risk register (R5) explicitly worries about, and it would make the cognitive
  plane's outputs unverifiable by construction.

## Consequences

- Verified end to end in Week 9: a `sensor_fault` incident auto-applies without pausing (its
  playbook is trivially safe), while an `accident` incident pauses at a real durable
  LangGraph checkpoint and resumes correctly via `Command(resume=...)` after producing a
  what-if-grounded proposal.
- Not every playbook type is simulatable yet: `run_what_if` gives a real, simulation-backed
  verdict for `switch_to_max_pressure` and `hold_current_plan`, and honestly reports
  `checked=False` for every other playbook type rather than fabricating an effect — a named,
  intentional gap, not a silently assumed one (see `tests/unit/test_what_if.py`'s
  `TestUnsimulatablePlaybooks`).
- This is also where R5 ("what-if state cloning is inaccurate") is meant to be answered
  honestly: the twin fork approximates state from detector-implied queues and the demand
  model, not true vehicle positions — the design's own stated mitigation is to publish the
  calibration error, which requires comparing what-if predictions to actual outcomes on
  scripted incidents, work not yet done in this build (see ADR-0013).
