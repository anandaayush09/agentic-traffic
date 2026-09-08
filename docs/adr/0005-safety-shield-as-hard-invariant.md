# ADR-0005: Safety shield as a hard invariant outside the learned policy

**Status:** Accepted (design Part 3; built Week 3 part 1, commit `bf949cc`)

## Context

A learned (RL) or LLM-influenced signal-control policy can, in principle, request an unsafe
phase transition — one that violates minimum/maximum green time, or an invalid phase for the
junction's actual signal program. Trusting the policy to have internalized these constraints
from reward shaping alone is both unverifiable and, for a safety-relevant system, not
credible.

## Decision

A `SignalShield` module — deliberately SUMO-agnostic, dependency-free, and exhaustively
tested with property-based testing (`hypothesis`) — sits between every policy (RL, max-
pressure, fixed-time, actuated) and the actual signal executor. It enforces minimum/maximum
green time and legal-phase constraints as hard invariants, independent of and downstream from
whatever chose the requested action. Action masking additionally encodes the shield *inside*
the RL policy's action distribution during training (invalid actions get −∞ logits), so
training and execution see identical constraints — the shield is a backstop, not a
first-time-encountered filter at deployment.

## Alternatives considered

- **Trust the policy** (rely on reward shaping alone to keep the RL policy within safe
  bounds): rejected as non-negotiable for credibility — an unshielded learned controller is
  not something a reviewer (or a real traffic engineer) should accept regardless of measured
  performance.

## Consequences

- The shield's exhaustive property-based test suite caught a real boundary-condition bug
  before it ever reached integration: an invalid phase request arriving exactly at
  `max_green_s` could silently bypass the max-green override — found by `hypothesis`, not by
  example-based tests, which is the argument for property-based testing on safety-critical
  invariants specifically.
- Because the shield is SUMO-agnostic, it is fast enough to test exhaustively (property-based
  tests run in milliseconds) and is reused unchanged as both the RL action mask source and
  the classical controllers' shared safety layer — one implementation serves every control
  mode, not a per-mode reimplementation.
- `SignalShield.force()` (added Week 7 for preemption) is a documented, narrow exception:
  it bypasses min/max-green but still refuses a non-green phase — preemption needs to
  override *timing*, never *legality*.
