# ATLAS — Agentic Traffic Learning & Adaptive Signaling

**Design Document v1.0 (architecture frozen 2026-07-11)**

## What this is

ATLAS is a production-grade, agentic-AI traffic management system for a simulated smart-city
district of ~25 signalized intersections. Collaborative AI agents autonomously manage traffic
signals to reduce congestion, waiting time, travel time, emergency response time, fuel
consumption, and CO₂ emissions, while remaining robust to rush hours, accidents, road closures,
construction, emergency vehicles, VIP movement, weather disruptions, public events, sensor
failures, and communication failures.

It is built as a flagship portfolio project to the engineering standard of a top-tier AI lab:
event-driven microservices, multi-agent reinforcement learning trained from scratch, an
LLM-based cognitive layer with grounded tool use, a full MLOps loop, and statistically rigorous
evaluation against established traffic-engineering baselines.

## The core thesis: two-loop control

The single most important architectural decision in ATLAS is the separation of control into two
loops with different latencies, different intelligence, and different failure modes:

- **Fast loop (sub-second → 5 s):** Multi-Agent PPO policies — implemented from scratch in
  PyTorch — choose signal phases at every intersection every 5 simulated seconds. A safety
  shield (action masking for minimum/maximum green and yellow enforcement) guarantees the RL
  can never emit an unsafe signal program. A watchdog degrades control gracefully:
  **RL → max-pressure → fixed-time** whenever inputs go stale or anomalous. Emergency-vehicle
  preemption is a deterministic green-wave overlay — a solved algorithm we do not ask RL to
  rediscover.
- **Slow loop (seconds → minutes):** A LangGraph-orchestrated cognitive layer — an Incident
  Commander (supervisor) and an Analyst — reasons over incidents at episodic cadence. Its
  tools are typed and grounded: query the digital twin's state, **run counterfactual what-if
  simulations on the twin**, apply a response playbook, request human operator approval.
  LLMs never sit in the real-time control path, and the system is fully functional with the
  LLM offline via deterministic playbooks.

This split is what makes the design defensible in a senior interview: it mirrors how real
safety-critical autonomy stacks are built (learned components inside deterministic envelopes),
and it makes every AI claim measurable.

## System at a glance

| Layer | What runs there |
|---|---|
| Digital twin | SUMO 1.27 (libsumo) on a real OpenStreetMap district; scenario engine with 10 disruption types; doubles as a counterfactual what-if service |
| Control plane | N Intersection Agents (shared-parameter MAPPO), Corridor Coordinators, deterministic Preemption Overlay, safety shield + fallback hierarchy |
| Cognitive plane | LangGraph Supervisor + Analyst; free-tier LLM (Groq / Gemini Flash) → Ollama → deterministic playbooks |
| ML services | Perception (YOLO on UA-DETRAC, ONNX INT8 edge profile), Forecasting (Graph WaveNet-class ST-GNN, 15/30/60-min horizons), Anomaly detection (EWMA + LSTM-autoencoder) |
| Backbone | Redpanda (Kafka API), Redis hot state, PostgreSQL + TimescaleDB, FastAPI services, schema-registered topics |
| UI / HITL | React + MapLibre operations dashboard; playbook approval queue; XAI panel with decision logs and counterfactuals |
| MLOps / deploy | MLflow, DVC, GitHub Actions, docker-compose (primary), Helm on kind, Prometheus + Grafana, per-hop latency SLOs |

## What makes it credible (and different from every RL-traffic repo on GitHub)

1. **MAPPO written from scratch** (CleanRL-style, ~500 lines) — not an RLlib config. Ray Core
   parallelizes SUMO rollout workers.
2. **Perception is load-bearing, not decorative:** the YOLO model's *measured* error
   distribution is injected into the twin's sensor streams, and control robustness to real
   perception noise is quantified.
3. **Statistical rigor:** every headline number comes from 10 seeded runs with 95% confidence
   intervals against three baselines (fixed-time, SUMO-actuated, max-pressure) across a
   10-scenario disruption suite.
4. **The LLM layer is evaluated, not vibes:** a scripted incident suite scores
   playbook-selection accuracy; the what-if tool grounds every recommendation in simulation.
5. **Operational honesty:** chaos tests (kill an agent, drop a partition), published p50/p99
   latency per hop, threat model, JWT-protected control APIs, and a read-only public demo.

## Constraints the design respects

- Solo engineer, 12 weeks × 20+ hrs/week (~250 hrs), new to RL/MARL and distributed systems —
  the roadmap front-loads ramp-up weeks and introduces infrastructure incrementally.
- Training compute is Colab/Kaggle only — training is CPU-bound on SUMO stepping, so the
  design mandates libsumo, 5-second decision intervals, ~25 intersections, parameter sharing,
  and fully resumable checkpointed runs.
- LLM budget is $0 — the cognitive layer runs at episodic cadence within free-tier limits
  (Groq llama-3.1-8b-instant: 14.4K req/day; Gemini 3 Flash: 1,500 req/day), with local Ollama
  and deterministic playbooks as fallbacks.
- Deployment is local-first (docker-compose), with one Helm chart exercised on kind and a
  free-tier hosted read-only replay demo for recruiters.

## Document map

| # | File | Contents |
|---|---|---|
| 1 | 01_problem_definition.md | Problem, challenges, existing solutions, gaps, success metrics, requirements |
| 2 | 02_literature_survey.md | SCATS, SCOOT, SURTRAC, MARL traffic control, ITS, prediction, digital twins |
| 3 | 03_system_architecture.md | Full production architecture and every design decision |
| 4 | 04_agent_architecture.md | Every agent and service, 11-field specification each |
| 5 | 05_communication.md | Protocols, JSON schemas, Kafka topics, coordination, fault tolerance |
| 6 | 06_ai_pipeline.md | CV, MARL, forecasting, GNN, optimization, anomaly detection |
| 7 | 07_tech_stack.md | Alternatives compared; every technology choice justified |
| 8 | 08_data_strategy.md | Maps, demand generation, datasets, sensor streams, versioning |
| 9 | 09_repository_design.md | Professional GitHub repository layout |
| 10 | 10_deployment.md | Docker, Kubernetes, CI/CD, monitoring, MLOps, security |
| 11 | 11_evaluation.md | Metrics, baselines, statistical protocol, chaos tests |
| 12 | 12_risks.md | Risk register, mitigations, challenged assumptions |
| 13 | 13_roadmap.md | 12-week implementation roadmap with resume bullets |
| 14 | 14_design_review.md | Multi-persona senior review: findings and resulting changes |
