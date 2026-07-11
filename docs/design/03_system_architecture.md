# Part 3 — System Architecture

## 3.1 Architectural overview

ATLAS is an event-driven microservice system organized into five planes around a central
event backbone. Everything communicates through schema-registered topics on Redpanda
(Kafka API); services are FastAPI/Python processes; state that must be fast is in Redis,
state that must be durable is in PostgreSQL/TimescaleDB.

```
                        ┌────────────────────────────────────────────────┐
                        │                COGNITIVE PLANE                 │
                        │  LangGraph Supervisor ("Incident Commander")   │
                        │  Analyst/XAI agent · Playbook registry         │
                        │  Tools: twin state · what-if sim · playbooks   │
                        │         operator approval                      │
                        └───────▲────────────────────────────┬───────────┘
                                │ incidents, forecasts       │ approved playbooks,
                                │ anomalies                  │ mode commands
┌───────────────┐       ┌───────┴────────────────────────────▼───────────┐
│   HITL / UI   │◀─────▶│              EVENT BACKBONE (Redpanda)         │
│ React+MapLibre│       │  topics: sensor.* vision.* signal.* control.*  │
│ approval queue│       │  incident.* forecast.* playbook.* audit.*      │
│ XAI panel     │       └───▲───────────▲───────────▲───────────▲────────┘
└───────────────┘           │           │           │           │
                    ┌───────┴───┐ ┌─────┴─────┐ ┌───┴─────┐ ┌───┴──────────┐
                    │ CONTROL   │ │ ML        │ │ DIGITAL │ │ OBSERVABILITY│
                    │ PLANE     │ │ SERVICES  │ │ TWIN    │ │ Prometheus   │
                    │ N Inter-  │ │ Perception│ │ SUMO    │ │ Grafana      │
                    │ section   │ │ Forecast  │ │ libsumo │ │ structured   │
                    │ Agents    │ │ Anomaly   │ │ scenario│ │ logs         │
                    │ Corridor  │ │           │ │ engine  │ │              │
                    │ Coords    │ │           │ │ what-if │ │              │
                    │ Preemption│ │           │ │ service │ │              │
                    │ Safety    │ │           │ │         │ │              │
                    │ shield    │ │           │ │         │ │              │
                    └───────────┘ └───────────┘ └─────────┘ └──────────────┘
                         Redis (hot state) · PostgreSQL+TimescaleDB (durable) · MLflow · DVC
```

## 3.2 The five planes and why they are separated

1. **Digital twin plane** — SUMO is the world. In development it *is* reality; in a real
   deployment it would run as a shadow twin. Isolating it behind event streams
   (`sensor.detections`, `signal.states` out; `control.commands` in) means nothing else in
   the system knows whether the world is simulated. **Decision rationale:** this is the
   sim-to-real seam; it makes every other component deployment-shaped.
2. **Control plane (fast loop)** — one logical Intersection Agent per junction, Corridor
   Coordinators over arterial groups, the deterministic Preemption Overlay, and the safety
   shield/watchdog. Runs at 5-simulated-second cadence, must never block on anything slower
   than Redis. **Rationale:** hard real-time work is isolated from everything with
   unpredictable latency (LLMs, DB writes, humans).
3. **ML services plane** — Perception, Forecasting, Anomaly Detection. Stateless consumers/
   producers on the bus with their own model lifecycles (MLflow). **Rationale:** models are
   deployed, versioned, and rolled back independently of control.
4. **Cognitive plane (slow loop)** — LangGraph graph with durable checkpointing; consumes
   `incident.*`, `forecast.*`, `anomaly.*`; produces `playbook.proposed` →
   (`operator approval`) → `playbook.apply`. **Rationale:** minutes-latency reasoning with
   external API dependencies must not share a failure domain with signal control.
5. **HITL/UI plane** — dashboard (read), approval queue and overrides (mutating, JWT).
   **Rationale:** the operator is a first-class actor, not an afterthought.

## 3.3 Cloud vs. edge architecture

ATLAS's processes map onto a realistic edge/cloud split, documented so the design is honest
about where each piece would physically run in a city:

| Tier | Real deployment | ATLAS realization |
|---|---|---|
| Edge (cabinet at intersection) | Signal controller + camera + detection model (INT8) + local fallback logic | Perception edge profile (ONNX INT8, latency benchmarked); safety shield + max-pressure fallback are pure-Python, no-dependency modules — exactly what would live in the cabinet |
| Near-edge / district hub | Intersection Agents, Corridor Coordinators, preemption | Control-plane containers |
| Cloud / TMC | Twin, training, forecasting, cognitive plane, dashboards, MLOps | Everything else in compose |

**Decision rationale:** we do not pretend to own roadside hardware. Instead the *software
boundary* is drawn where the hardware boundary would be: everything below the bus keyed by
`intersection_id` could be packed into a cabinet image. The edge story is demonstrated by (a)
the quantized perception profile with measured latency, and (b) the fallback ladder running
without network connectivity (local decision from local detector state).

## 3.4 Sensors and cameras

- **Simulated sensors (from SUMO):** lane-area detectors (queue length, occupancy), induction
  loops (counts, speeds), and signal state — emitted to `sensor.detections` /
  `signal.states` at 1 Hz. A **sensor fault model** (configurable dropout, bias, noise,
  freeze) sits between SUMO and the bus; its noise parameters can be set from the measured
  YOLO error profile (Part 6).
- **Camera path:** the Perception service consumes recorded traffic-camera video
  (UA-DETRAC clips as stand-in feeds), runs YOLO, and emits the *same*
  `vision.detections` schema the simulated detectors use. **Rationale:** identical schema
  is the point — control cannot tell detection source apart, which is both the sim-to-real
  argument and the mechanism for the robustness experiments.

## 3.5 Databases

| Store | Technology | Holds | Why |
|---|---|---|---|
| Hot state | Redis | Latest per-intersection observation vectors, signal states, agent health heartbeats, preemption locks | Fast loop reads must be sub-ms; TTL'd keys make staleness detection trivial (watchdog reads key age) |
| Telemetry | PostgreSQL + TimescaleDB | All bus events (hypertables partitioned by time), metrics per decision cycle | SQL for evaluation queries; Timescale continuous aggregates power the dashboard's historical charts |
| Audit | PostgreSQL | Decision log: input snapshot hash, action, shield delta, policy version, explanation id | NFR auditability; append-only |
| Artifacts | MLflow (local backend) + DVC remote | Models, checkpoints, datasets, scenario definitions | Standard MLOps separation of code (git), data (DVC), models (MLflow) |

**Decision rationale:** one relational engine (Postgres) with Timescale beats adding a
dedicated TSDB (InfluxDB) — fewer moving parts for a solo engineer, and recruiters recognize
Postgres. Redis is justified because the fast loop's read pattern (latest-value by key, TTL
staleness) is exactly Redis's sweet spot; using Kafka alone would force every agent to
maintain consumer state for data it only needs the newest value of.

## 3.6 APIs

All services expose FastAPI apps with OpenAPI docs; only three have mutating endpoints:

- **Control API** (`/intersections/{id}/mode`, `/overrides`, `/preemption/trigger`) — JWT,
  operator-only; every call lands in the audit log.
- **Cognitive API** (`/playbooks/{id}/approve|reject`, `/incidents/{id}`) — JWT; the approval
  queue backend.
- **Twin API** (`/whatif` — clone state, apply scenario delta, fast-forward N sim-minutes,
  return metric deltas; `/scenario/load`) — internal-only; the supervisor's grounding tool.

Read-only APIs (dashboard queries, replay, metrics) are unauthenticated locally,
and are the only surface exposed by the public demo.

## 3.7 Event bus and message queue

Redpanda, one broker locally (Kafka wire protocol — clients, tooling, and resume keyword are
Kafka; the doc's Part 7 carries the full comparison). Topic taxonomy, partitioning (keyed by
`intersection_id` so per-intersection ordering is guaranteed), schemas, delivery semantics,
retries, and DLQs are specified in Part 5. Design rules:

- The bus is the **only** inter-plane communication path (no service-to-service RPC except
  the twin's what-if API from the supervisor, and Redis reads inside the fast loop).
- Events are **facts, not commands**, except the explicit `control.commands` and
  `playbook.apply` topics.
- **Event sourcing:** replaying `sensor.*`+`signal.*`+`control.*` reproduces any run — this
  powers the replay dashboard and debugging.

## 3.8 Digital twin service

Three run modes from one codebase:
1. **Live mode** — the world for the demo: SUMO stepping in real time (or scaled), publishing
   sensor/signal events, consuming commands.
2. **Training mode** — headless libsumo environments inside Ray rollout workers (no bus, no
   services — pure in-process env for throughput; same wrapper code, different transport).
3. **What-if mode** — on request: fork current network state (vehicle positions approximated
   from detector state + demand model), apply hypothetical (close lane, reroute, change
   plan), fast-forward 15–30 sim-minutes at max speed, return metric deltas.

**Decision rationale:** the training/serving split (in-process vs. bus-attached) is the only
way to get RL sample throughput on Colab while keeping the production path event-driven; the
shared env wrapper guarantees the policy sees identical observation semantics in both.

## 3.9 ML services, monitoring, dashboards

- **ML services** each follow one pattern: consume topic → infer → produce topic, with
  model loaded from MLflow registry by stage (`Production`/`Staging`), Prometheus metrics
  (inference latency, input rate, drift proxies), health endpoint.
- **Monitoring:** Prometheus scrapes every service; Grafana dashboards: (1) traffic KPIs,
  (2) system latency per hop, (3) agent behavior (actions, rewards, fallback engagements),
  (4) LLM layer (calls, tokens, cache hits, playbook outcomes).
- **Operations dashboard (React + MapLibre):** live map (signal states, queues, incidents,
  emergency vehicles), forecast overlays, approval queue, XAI panel (per-decision
  explanations, what-if results), replay scrubber.

## 3.10 Design decisions summary (the "why" table)

| Decision | Alternatives rejected | Reason |
|---|---|---|
| Two-loop control, LLM out of fast path | LLM-in-the-loop control | Latency, cost, safety, and honesty: no LLM meets a 1 s p99 under free-tier limits; learned+shielded fast loop mirrors real autonomy stacks |
| Event bus as sole integration | REST mesh, shared DB | Replay/audit for free; decoupled failure domains; the distributed-systems learning goal |
| Redpanda (Kafka API) | Apache Kafka+ZK, RabbitMQ, Redis Streams | Kafka semantics + one-binary local footprint; RabbitMQ lacks replay; details in Part 7 |
| Redis for hot state | Kafka-only, in-proc state | Latest-value + TTL staleness pattern; watchdog simplicity |
| Postgres+Timescale | InfluxDB, ClickHouse | One engine, SQL evaluation queries, hiring-manager legibility |
| SUMO + libsumo | CityFlow, Vissim, CARLA | OSM import, emissions model, PT support, ecosystem; libsumo kills the socket bottleneck; CityFlow faster but no emissions/PT and dead ecosystem |
| Safety shield + degradation ladder in control plane | Trust the policy | Non-negotiable for credibility; also the best interview material in the project |
| Twin-as-tool for the supervisor | LLM free-form reasoning | Grounds the cognitive plane in simulation, converts "agentic AI" from claim to measured capability |
