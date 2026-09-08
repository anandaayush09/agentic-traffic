# Architecture Decision Records

One-page records of ATLAS's frozen architectural decisions, per
[09_repository_design.md](../design/09_repository_design.md)'s own convention: "each frozen
decision... gets a one-page ADR; interviewers love them and they are cheap because Parts 3–7
[of the design doc] already contain the content." These ADRs distill that content into the
standard Context / Decision / Alternatives / Consequences shape, plus two ADRs (0012, 0013)
that record decisions made during the actual 11-week build rather than in the upfront design.

| # | Title | Status | Week decided |
|---|---|---|---|
| [0001](0001-two-loop-control-llm-out-of-fast-path.md) | Two-loop control: LLM stays out of the fast path | Accepted | Design (Part 3) |
| [0002](0002-event-driven-architecture-redpanda-bus.md) | Event-driven architecture on a Redpanda (Kafka-API) bus | Accepted | Design (Part 3) / built Week 5 |
| [0003](0003-state-tiering-redis-hot-postgres-timescale-durable.md) | State tiering: Redis for hot state, Postgres+Timescale for durable state | Accepted | Design (Part 3) |
| [0004](0004-sumo-libsumo-digital-twin.md) | SUMO + libsumo as the digital twin | Accepted | Design (Part 3) / built Week 1 |
| [0005](0005-safety-shield-as-hard-invariant.md) | Safety shield as a hard invariant outside the learned policy | Accepted | Design (Part 3) / built Week 3 |
| [0006](0006-twin-as-tool-for-cognitive-plane-grounding.md) | The twin as a tool call, not free-form LLM reasoning | Accepted | Design (Part 3) / built Week 9 |
| [0007](0007-algorithmic-corridor-coordination-not-learned.md) | Algorithmic (not learned) corridor coordination | Accepted | Design (Part 4) / built Week 7 |
| [0008](0008-deterministic-preemption-overlay-not-rl.md) | Deterministic state machine for emergency preemption, not RL | Accepted | Design (Part 4) / built Week 7 |
| [0009](0009-mappo-from-scratch-not-rllib.md) | MAPPO implemented from scratch instead of RLlib | Accepted | Design (Part 6) / built Week 5 |
| [0010](0010-llm-provider-chain-with-deterministic-terminal-fallback.md) | LLM provider chain with a deterministic terminal fallback | Accepted | Design (Part 4) / built Week 9 |
| [0011](0011-playbooks-as-bounded-templates.md) | Playbooks as bounded, pre-modeled response templates | Accepted | Design (Part 4) / built Week 9 |
| [0012](0012-static-frozen-snapshot-public-demo.md) | Static frozen-snapshot public demo instead of a live backend | Accepted | Week 10 |
| [0013](0013-r1-r2-r4-descope-outcomes.md) | R1/R2/R4 risk-register outcomes, as actually realized | Accepted | Week 11/12 |
