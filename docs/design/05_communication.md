# Part 5 — Communication Architecture

## 5.1 Principles

1. **Bus-only integration.** All inter-plane communication is events on Redpanda (Kafka
   API). The two exceptions are deliberate: fast-loop Redis reads (latency), and the
   Supervisor→Twin what-if HTTP call (request/response semantics genuinely fit RPC).
2. **Events are facts.** Past-tense records (`sensor.detections`, `incident.detected`).
   Imperatives exist only on `control.commands`, `control.preemption`, `playbook.apply`.
3. **Single writer.** Every topic has exactly one producer type; every signal has exactly
   one commander at a time (enforced by preemption locks and mode state).
4. **Schema or it didn't happen.** Every topic has a versioned JSON Schema in-repo
   (`schemas/`), validated at produce and consume; unknown/invalid messages go to a DLQ.

## 5.2 Topic taxonomy

| Topic | Key | Producer → Consumer | Cadence |
|---|---|---|---|
| `sensor.detections.v1` | intersection_id | Twin adapter → state writer, anomaly, telemetry | 1 Hz × lanes |
| `vision.detections.v1` | camera_id | Perception → state writer, telemetry | per-frame batch |
| `signal.states.v1` | intersection_id | Twin adapter → dashboard, telemetry, coordinator | on change |
| `control.commands.v1` | intersection_id | Intersection Agents / Preemption → twin adapter, audit | per decision |
| `control.preemption.v1` | intersection_id | Preemption Overlay → agents, twin adapter, dashboard | per lock event |
| `corridor.hints.v1` | corridor_id | Coordinators → state writer (as features), dashboard | 60 s |
| `forecast.link_state.v1` | link_id | Forecasting → supervisor, dashboard | 5 min |
| `anomaly.events.v1` | intersection_id | Anomaly svc → supervisor, dashboard | on detection |
| `incident.events.v1` | incident_id | Supervisor / scenario engine → all planes | on change |
| `scenario.events.v1` | scenario_id | Scenario engine → all planes | scripted |
| `playbook.lifecycle.v1` | incident_id | Supervisor → control plane, dashboard, audit | on transition |
| `audit.decisions.v1` | intersection_id | All actors → Postgres sink | per decision |
| `system.health.v1` | service_id | All services → watchdog, Prometheus sink | 5 s heartbeat |

Partitioning: 12 partitions on hot topics, keyed as above — **per-key ordering** is the
correctness requirement (commands for one junction must apply in order; cross-junction
ordering is irrelevant). Retention: hot topics 24 h (replay window); `audit.*` and
`incident.*` compacted + mirrored to Postgres for permanence.

## 5.3 Representative JSON Schemas

`control.commands.v1`:

```json
{
  "$id": "atlas.control.commands.v1",
  "type": "object",
  "required": ["msg_id", "ts", "intersection_id", "actor", "action", "policy_version"],
  "properties": {
    "msg_id":         {"type": "string", "format": "uuid"},
    "ts":             {"type": "string", "format": "date-time"},
    "intersection_id":{"type": "string", "pattern": "^J[0-9]{3}$"},
    "actor":          {"enum": ["rl_agent", "max_pressure", "fixed_time", "preemption", "operator"]},
    "action": {
      "type": "object",
      "required": ["kind"],
      "properties": {
        "kind":     {"enum": ["set_phase", "extend_phase", "set_program"]},
        "phase_id": {"type": "integer", "minimum": 0},
        "duration_s": {"type": "number", "minimum": 0}
      }
    },
    "shield": {
      "type": "object",
      "properties": {
        "modified": {"type": "boolean"},
        "reason":   {"enum": ["min_green", "max_green", "yellow_transition", "preemption_lock", null]}
      }
    },
    "policy_version": {"type": "string"},
    "priority":       {"enum": ["normal", "preemption"], "default": "normal"}
  }
}
```

`incident.events.v1` (abridged):

```json
{
  "$id": "atlas.incident.events.v1",
  "type": "object",
  "required": ["msg_id", "ts", "incident_id", "class", "status", "location", "confidence"],
  "properties": {
    "incident_id": {"type": "string"},
    "class":  {"enum": ["accident", "closure", "construction", "weather", "event_surge",
                         "sensor_fault", "comms_fault", "emergency_run", "vip_run", "unknown"]},
    "status": {"enum": ["detected", "triaged", "responding", "monitoring", "cleared"]},
    "location": {"type": "object", "properties": {
        "link_ids": {"type": "array", "items": {"type": "string"}},
        "geo": {"type": "array", "items": {"type": "number"}}}},
    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    "evidence":   {"type": "array", "items": {"type": "string"},
                    "description": "msg_ids / what-if run ids supporting this state"}
  }
}
```

All schemas share an envelope (`msg_id`, `ts`, `schema_version`, `trace_id`) — `trace_id`
propagates from originating sensor event through command to audit, enabling the per-hop
latency measurement (§5.7).

## 5.4 Negotiation and coordination

- **Intersection ↔ Corridor:** advisory-hint pattern, not negotiation — coordinators publish
  offset/summary features; agents' policies weigh them (learned during training with
  hint-dropout). No agent can command another.
- **Preemption vs. control:** lock protocol via Redis `SETNX` + TTL. Lock holder is the only
  legal producer of commands for that junction (twin adapter enforces: commands with
  `priority=normal` are dropped while a lock exists — enforcement at the *consumer*, so a
  buggy or restarted agent cannot violate preemption).
- **Conflicting emergency runs:** priority class ordering (emergency > VIP), then
  first-lock-wins per junction; the overlay re-times the losing route around the contested
  junctions. Starvation bound documented (VIP waits at most one clearance interval per
  junction).
- **Supervisor vs. operator:** proposals above the impact threshold require approval;
  approvals are per-proposal (no blanket grants); a rejected proposal returns the incident
  to triage with the rejection as context.

## 5.5 Distributed consensus — why ATLAS doesn't hand-roll it

There is no state in ATLAS requiring multi-writer agreement: signals have single writers
(locks arbitrate), topics give per-key total order, and Redis (single instance locally) is
the lock authority. Hand-rolling Raft would add risk without a requirement — the honest
senior answer. The document's city-scale note: at multi-district scale you would run
replicated Redis (Redlock caveats acknowledged — TTL'd locks over a single authority per
district remain preferable), Redpanda's internal Raft already replicates the log, and the
one true consensus need (district-level configuration changes) rides on Postgres.

## 5.6 Delivery semantics, retries, fault tolerance

- **Producers:** acks=all, idempotent producer on; retries with exponential backoff + jitter
  (100 ms → 5 s, capped); on exhaustion → local disk spool + `system.health` degradation.
- **Consumers:** at-least-once; all consumers idempotent via `msg_id` dedup (Redis set with
  TTL ≥ retention). Poison messages → 3 attempts → DLQ topic + alert.
- **Command staleness:** every command carries `ts`; the twin adapter drops commands older
  than 2 decision cycles (a late command is a wrong command) — dropped-stale is a metric.
- **Comms-failure scenario (FR2):** the scenario engine can partition topics/pause consumers
  per district area; expected behavior is the fallback ladder within 2 cycles and recovery
  without operator action — verified by chaos tests, not asserted.

## 5.7 Latency budget (fast loop, detection → actuation)

| Hop | Budget p99 |
|---|---|
| Twin adapter publish → broker | 20 ms |
| State-writer consume → Redis write | 50 ms |
| Agent read + policy forward + shield | 150 ms |
| Command publish → broker | 20 ms |
| Twin adapter consume → actuation | 60 ms |
| Slack (GC, scheduling, retries) | 700 ms |
| **Total** | **< 1 s p99** (vs. 5 s decision interval → 5× headroom) |

Measured continuously via `trace_id` spans exported to Prometheus histograms; the published
report includes p50/p99 per hop under load and under chaos (Part 11).
