# ADR-0008: Deterministic state machine for emergency preemption, not RL

**Status:** Accepted (design Part 4 §4.3; built Week 7 parts 3–4, commits `cc7cc7a`,
`374fe5f`)

## Context

Emergency-vehicle preemption (and VIP movement, the same mechanism at lower priority) is
safety-critical, comparatively rare, and has a well-understood solved form in real traffic
engineering: compute ETAs along the route, lock each junction to green just ahead of arrival,
release behind the vehicle.

## Decision

A dedicated `PreemptionOverlay` runs a deterministic state machine (`PLANNED → LOCKING →
ACTIVE → RELEASING → RECOVERY`) driven by an explicit simulation-time parameter (`now_s`),
not RL. Junction locks are acquired via an atomic SETNX-with-TTL pattern
(`InMemoryLockStore`, standing in for Redis) so a lock the overlay fails to release (e.g. the
process dies mid-run) self-expires rather than stranding the junction under permanent
override.

## Alternatives considered

- **Learn preemption via RL:** rejected outright — preemption is safety-critical, rare (too
  little training signal for RL to reliably specialize on), and already has a correct
  deterministic solution. Learning it would be, in the design doc's own words, "negligent
  engineering" — spending model capacity and risk budget on a solved problem.

## Consequences

- The TTL self-expiry mechanism is directly chaos-testable without needing to actually kill a
  process: `tests/chaos/test_kill_preemption_overlay.py` narrates lock-acquired →
  overlay-dies-holding-it → TTL-elapses → fresh-run-recovers-the-junction as one incident
  timeline against a fake clock, and it passes deterministically (no flaky timing test).
- `now_s` (simulation time, used for ETA/lock-scheduling logic) and the LockStore's TTL
  (always real wall-clock, since "the overlay process died" is a real-world event) are two
  intentionally different clocks — documented explicitly in `preemption.py` so a future
  reader doesn't assume they should be unified.
- Measured, not simulated: a real two-seed evaluation (Week 7 part 4,
  `scripts/eval_emergency_preemption.py`) showed emergency-vehicle response time dropping
  from 40.0s to 25.0s (37.5%) with the overlay engaged along a real 5-junction corridor —
  reported as a single-decision-granularity measurement, with the full multi-seed statistical
  study left to Week 11's evaluation suite.
