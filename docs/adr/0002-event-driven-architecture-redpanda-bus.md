# ADR-0002: Event-driven architecture on a Redpanda (Kafka-API) bus

**Status:** Accepted (design Part 3/5; built Week 5 part 5, commit `0bc6ca7`)

## Context

ATLAS is ~12 services across 5 planes (twin, control, ML services, cognitive, HITL/UI). They
need to communicate without becoming a tightly coupled monolith, and the project's own
evaluation/replay/audit requirements ("reproduce any run," "decision log," "explain why
junction 12 switched plans") need a durable record of what happened, not just current state.

## Decision

All inter-plane communication is events on a bus (no service-to-service RPC), with two
explicit, documented exceptions: fast-loop Redis reads (latency-critical, latest-value
semantics don't fit event streams well) and the Supervisor's what-if call to the twin
(genuinely request/response). Redpanda was chosen as the specific broker: Kafka wire
protocol, single-binary local footprint, real replay/retention semantics.

Every topic has a versioned JSON Schema in `schemas/`, validated at both produce and consume;
`BusProducer`/`BusConsumer` depend on their raw client only structurally (a Python
`Protocol`), so real `kafka-python` objects and test fakes both work with no adapter class.

## Alternatives considered

- **REST mesh / shared database:** rejected — no replay/audit for free, and failure domains
  couple directly (a slow consumer's DB query can block a producer's write path).
- **Apache Kafka + ZooKeeper:** rejected for local-dev footprint; Redpanda gives the same
  wire protocol and tooling without a second coordination service.
- **RabbitMQ:** rejected — lacks the replay semantics (durable, offset-addressable log) the
  design's event-sourcing/replay-dashboard requirement depends on.
- **`confluent-kafka` client** over `kafka-python`: rejected specifically because this
  project had already hit three separate native-DLL collisions on Windows during
  development (SB3/torch, libsumo-in-Ray, libsumo-vs-mlflow import ordering) — a
  pure-Python Kafka client avoided a fourth.

## Consequences

- Every producer/consumer pair is independently unit-testable with no broker running (21 of
  Week 5's new tests are broker-free); exactly one integration test
  (`tests/integration/test_bus_roundtrip.py`) needs a live broker and self-skips via a
  reachability check when none is present — the honest, documented state of this repo on a
  machine without Docker.
- `control.commands.v1`'s `intersection_id` pattern deliberately deviates from the design
  doc's illustrative `^J[0-9]{3}$` regex to match real SUMO `tls_id`s (e.g.
  `GS_42439981`) — a concrete instance of the design doc's own "schemas are Representative,
  not literal contracts" caveat (05_communication.md §5.3).
- Week 11's chaos testing (`tests/chaos/test_partition_bus.py`) found and fixed a real gap
  this architecture makes possible to test cleanly: a bus partition (producer exists, every
  `send()` fails) must not crash the control loop, since publishing is documented as
  best-effort, not load-bearing for the decision itself — `IntersectionAgentService` had no
  exception handling around its publish calls until this was caught.
