# Part 4 — Agent Architecture

## 4.0 Taxonomy (deliberate, and worth defending in interviews)

ATLAS refuses "agent inflation." An *agent* perceives, decides, and acts with some autonomy;
a *service* transforms data; a *scenario event* is just a fact on the bus. The original
12-agent wish list maps as follows:

| Requested "agent" | ATLAS realization |
|---|---|
| Intersection Agent | **Agent** — RL policy per junction (§4.1) |
| Corridor Agent | **Agent** — algorithmic coordinator (§4.2) |
| Emergency Vehicle Agent | **Agent** — deterministic Preemption Overlay (§4.3) |
| Supervisor Agent | **Agent** — LLM Incident Commander (§4.4) |
| Traffic Police / Human ops | **Human**, via HITL console; the Analyst agent briefs them (§4.5) |
| Learning Agent | **Offline system** — the training pipeline (§4.6) |
| Traffic Prediction Agent | **Service** — Forecasting (§4.7) |
| Accident Detection Agent | **Service** — Anomaly Detection (§4.8) |
| Resource Allocation Agent | **Function of the Supervisor** — playbooks allocate green-time budgets and reroutes (§4.4) |
| Weather Agent | **Scenario events** (`scenario.weather`) + playbooks; a process that forwards a weather API is not an agent |
| Construction Agent | **Scenario events** (`scenario.closure`) + playbooks |
| Public Transport Agent | **Policy feature + playbooks**: bus priority enters the Intersection Agent's observation/reward; PT disruptions are events (§4.1, §4.4) |

Each agent/service below is specified on the same 11 fields: responsibilities, inputs,
outputs, memory, reasoning, planning, communication, APIs, decision-making, failure recovery,
scalability.

---

## 4.1 Intersection Agent (×25, the fast loop's core)

- **Responsibilities.** Select the signal phase for one junction every 5 simulated seconds to
  minimize local pressure and network delay; honor the safety shield; grant bus priority when
  requested; yield to preemption locks; report health.
- **Inputs.** Per-lane queue lengths, occupancy, mean speed (from Redis latest-state, written
  by the sensor stream consumer); current phase + elapsed time; neighbor outflow summaries
  (from Corridor Coordinator); bus-priority requests; preemption lock flag.
- **Outputs.** `control.commands` events: `{intersection_id, action: phase_id | extend,
  policy_version, shield_applied, ts}`; per-decision audit record; Prometheus counters.
- **Memory.** Stateless between decisions except a short observation stack (last 3
  observations) encoded in the input; no long-term memory — deliberately, so failover =
  respawn.
- **Reasoning.** Shared-parameter MAPPO actor (Part 6): MLP over the normalized observation;
  action masking from the shield *inside* the policy's action distribution (invalid actions
  get −∞ logits), so training and execution see identical constraints.
- **Planning.** None explicitly — the value function learned under CTDE internalizes
  downstream consequences; corridor-level lookahead is the coordinator's job.
- **Communication.** Consumes Redis state + coordinator hints; produces command events.
  Never calls another agent synchronously.
- **APIs.** `/health`, `/mode` (rl | max_pressure | fixed | off — operator-settable, JWT).
- **Decision-making.** Every 5 s: read state → staleness check → (if stale > 2 cycles →
  fallback ladder) → policy forward pass → shield → emit command.
- **Failure recovery.** Watchdog pattern: if the agent process dies, its Redis heartbeat key
  expires; a supervisor process restarts it; meanwhile the twin-side controller holds the
  last program and, after 2 missed cycles, the junction's local fallback (max-pressure
  module co-located with the twin adapter) takes over. Recovery is a measured chaos test.
- **Scalability.** Agents are partitioned across worker processes (25 fit in one; the design
  shards by `intersection_id` hash across K workers). Parameter sharing means one model
  serves all junctions; per-junction geometry is handled by observation normalization and
  action masks, the MPLight-proven approach. Documented path to 500 junctions: more shards,
  same model.

## 4.2 Corridor Coordinator (×3–5, one per arterial group)

- **Responsibilities.** Provide the coordination signal MARL alone struggles with at longer
  ranges: compute green-wave offset recommendations along its arterial from current platoon
  measurements; summarize corridor state for member intersections; detect corridor-level
  spillback and request upstream metering.
- **Inputs.** Member intersections' detector states and current phases; forecast snapshots
  (15-min) for anticipatory offsets; scenario events affecting its links.
- **Outputs.** `corridor.hints` events: per-member neighbor-summary vectors (consumed as
  policy observation features) and offset recommendations; `corridor.alerts` on spillback.
- **Memory.** Rolling 15-minute window of link travel times and platoon detections.
- **Reasoning.** Algorithmic, not learned: classic bandwidth-maximization (green wave)
  offsets from measured platoon speeds + a spillback detector (occupancy > threshold on exit
  links). **Why not learned:** corridor coordination via learned communication is a research
  project in itself; algorithmic hints keep the system explainable and give the RL layer a
  curriculum-stable feature. (Documented as a stretch: replace with learned GAT messages à la
  CoLight and A/B the two.)
- **Planning.** Rolling-horizon offset recomputation every 60 s.
- **Communication.** Bus only; hints are advisory features, never commands — intersections
  remain the single writer of their own signals (single-writer principle).
- **APIs.** `/health`, `/corridor/{id}/state` (read).
- **Decision-making.** Deterministic recomputation on timer + event-triggered on alerts.
- **Failure recovery.** Stateless recompute from the telemetry window on restart; if absent,
  intersections simply lose the hint features (policy trained with hint-dropout so this
  degrades smoothly — another deliberate robustness experiment).
- **Scalability.** One coordinator per corridor; corridors are configuration, not code.

## 4.3 Emergency Preemption Overlay (singleton per district)

- **Responsibilities.** On an emergency run (`scenario.emergency` or operator trigger),
  compute the route's intersection sequence and ETAs, acquire preemption locks junction by
  junction just-in-time, command green waves, release locks behind the vehicle, and hand
  control back to RL with a recovery marker (so recovery time is measurable).
- **Inputs.** Emergency vehicle route + live position (from twin/GPS events); signal states.
- **Outputs.** `control.preemption` lock/unlock events; direct phase commands during the
  lock; `audit.*` records; recovery-start markers.
- **Memory.** Active runs table (in Redis, TTL'd).
- **Reasoning.** Deterministic algorithm: ETA per junction from live speed; lock at
  ETA − (clearance time); phase to the approach's green with standard transition; VIP
  movement is the same mechanism with a policy flag (lower priority class, no sirens
  assumption). **Why not RL:** preemption is safety-critical, rare, and already solved;
  learning it would be negligent engineering — the interview-ready answer.
- **Planning.** Route-ahead lock scheduling; re-plans on vehicle deviation.
- **Communication.** Locks via Redis (atomic SETNX with TTL — a lock leak self-heals);
  commands via the same `control.commands` topic with an override priority field.
- **APIs.** `/preemption/trigger` (JWT, operator/dispatch), `/preemption/active`.
- **Decision-making.** Event-driven state machine per run: PLANNED → LOCKING → ACTIVE →
  RELEASING → RECOVERY.
- **Failure recovery.** TTL'd locks guarantee junctions return to normal control even if the
  overlay dies mid-run; watchdog restarts and reconstructs from the active-runs table.
- **Scalability.** Runs are independent; conflicting routes resolved by priority class +
  first-lock-wins with documented starvation limits.

## 4.4 Supervisor Agent — "Incident Commander" (LLM, slow loop)

- **Responsibilities.** Triage incidents and anomalies; select and parameterize response
  playbooks; verify candidate playbooks in the twin's what-if service; submit for operator
  approval when disruptive; apply approved playbooks; monitor outcomes; stand down responses.
- **Inputs.** `incident.*`, `anomaly.*`, `forecast.*`, `corridor.alerts`, operator messages;
  playbook registry; twin state via tool.
- **Outputs.** `playbook.proposed` (with rationale + what-if evidence), `playbook.apply`
  (post-approval), `incident.status`, operator-facing briefs.
- **Memory.** LangGraph durable checkpoints (survives restart mid-incident — LangGraph 1.2's
  durable execution is exactly this); an incident journal in Postgres; a semantic cache of
  (incident-signature → playbook decision) to conserve free-tier tokens.
- **Reasoning.** LLM (provider chain: Groq llama-3.1-8b-instant / Gemini 3 Flash free tier →
  local Ollama → **no-LLM deterministic mapper**) with structured JSON output; every claim
  about consequences must cite a what-if run id — the graph enforces tool-grounding by
  construction (a proposal node cannot be reached without a verification node).
- **Planning.** Playbooks *are* the plans: parameterized, pre-modeled response templates
  (e.g., `divert_arterial(A→B, duration)`, `meter_upstream(corridor, rate)`,
  `bus_bridge(route)`, `weather_capacity_derate(area, factor)`). The LLM's job is selection,
  parameterization, sequencing, and stand-down — not free-form control. This mirrors real
  TMC practice and bounds the blast radius of model error.
- **Communication.** Bus consumer/producer; the *only* component that calls the twin's
  what-if API; never writes `control.commands` directly — playbooks compile to mode/limit
  changes that the control plane interprets.
- **APIs.** `/incidents`, `/playbooks/{id}/approve|reject` (JWT — the approval queue).
- **Decision-making.** LangGraph state machine: CLASSIFY → RETRIEVE_PLAYBOOKS →
  PARAMETERIZE → VERIFY(what-if) → {AUTO_APPLY if low-impact | REQUEST_APPROVAL} → MONITOR →
  STAND_DOWN. Impact thresholds (which playbooks auto-apply) are configuration, reviewed in
  the threat model.
- **Failure recovery.** Durable graph checkpoints; if the LLM chain is exhausted/down, the
  deterministic mapper (incident class → default playbook) keeps responses flowing — the
  system's value degrades from "reasoned" to "scripted," never to "absent." All free-tier
  rate limits enforced client-side with token buckets.
- **Scalability.** Incidents are independent graph instances; concurrency bounded by
  provider rate limits (measured: a demo day consumes < 200 calls, well inside Groq's 14.4K
  req/day and Gemini's 1,500 req/day free quotas).

## 4.5 Analyst / XAI Agent (LLM, slow loop)

- **Responsibilities.** Turn decision logs into operator-readable explanations; answer
  operator questions ("why did junction 12 switch plans at 8:42?"); produce post-incident
  reports and post-episode performance briefs; flag policy behavior drift.
- **Inputs.** Audit log, decision snapshots, what-if results, evaluation metrics, operator
  queries from the dashboard.
- **Outputs.** Explanation records keyed to decisions (XAI panel), incident postmortems
  (markdown), weekly performance briefs.
- **Memory.** Retrieval over the audit store (SQL + embeddings over incident journals).
- **Reasoning.** LLM with strict grounding: explanations must quote the logged observation
  values and shield actions; numeric claims are computed by tools (SQL templates), not by
  the model. Where the question is "why this action," the answer combines (a) logged policy
  inputs, (b) counterfactual replays ("with the queue on the north approach halved, the
  policy chooses phase 2"), which the agent can request from the twin.
- **Planning.** None — request/response and scheduled briefs.
- **Communication.** Bus consumer; dashboard API.
- **APIs.** `/explain/{decision_id}`, `/ask` (JWT for ask; explain is read-only).
- **Decision-making.** Template-guided generation with tool calls; refuses (states
  insufficiency) when logs don't cover the question — hallucinated explanations are worse
  than none, and the eval suite (Part 11) scores refusal correctness too.
- **Failure recovery.** Stateless; explanations are best-effort and queue-backed.
- **Scalability.** Fully parallel; cache by decision id.

## 4.6 Learning system (offline "Learning Agent")

Not a runtime agent: the training pipeline (Part 6) — Ray rollout workers around libsumo
envs, the from-scratch MAPPO trainer, curriculum stages, MLflow-tracked runs, checkpoint
promotion via the registry (Staging → shadow evaluation on the disruption suite → Production).
Policy updates reach the control plane by registry stage change + agent hot-reload; rollback
is a stage flip. Continuous learning from live operation is explicitly out of scope (offline
retraining on accumulated telemetry is the documented path) — stated to preempt the
"you can't do online RL on a city" review.

## 4.7 Forecasting Service

Consume `sensor.*` aggregates (5-min bins) → Graph WaveNet-class ST-GNN → publish
`forecast.link_state` for 15/30/60-min horizons every 5 min, with per-horizon uncertainty
(quantile heads). Inputs also include time-of-day/day-of-week encodings and active scenario
flags (weather derate, closures) so forecasts remain meaningful under disruption. Model from
MLflow registry; drift monitor compares rolling MAE to training MAE and emits
`anomaly.model_drift` (consumed by the Supervisor). Failure = stale forecast topic; consumers
treat forecasts as advisory, so staleness degrades anticipation, never control. Scales by
graph partitioning (documented, not needed at 25 nodes).

## 4.8 Anomaly Detection Service

Two-tier detector on detector streams: (1) EWMA/z-score per lane for fast, explainable
deviations (sudden occupancy spikes, flatlined sensors — sensor-fault signatures); (2) an
LSTM-autoencoder over multivariate per-junction windows for subtler patterns (incident
signatures: upstream occupancy up + downstream flow down). Emits typed `anomaly.*` events
with confidence and the contributing features (explainability requirement). Tuned for high
recall with Supervisor-side disambiguation (what-if + camera check playbook), because a
missed incident costs more than a false triage. Baseline tier keeps working when the ML tier
is down — same degradation philosophy as control.

## 4.9 Perception Service

YOLO (fine-tuned on UA-DETRAC) over camera streams → `vision.detections` (same schema as
simulated detectors). Two profiles: full-precision (cloud) and INT8 ONNX (edge), both
benchmarked (mAP, p50/p99 latency, throughput). Its second product is the **measured error
model** (per-class miss rate, false-positive rate, count noise vs. occlusion level) that
parameterizes the twin's sensor fault model — the bridge that makes the robustness study
honest. Failure: stream marked degraded; affected junctions' observations flagged, watchdog
weighs staleness accordingly.
