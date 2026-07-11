# Part 7 — Technology Stack

Every row states the decision, the serious alternatives, and why the winner wins *for this
project's constraints* (solo engineer, 12 weeks, Colab/Kaggle compute, $0 LLM budget,
local-first deployment, hiring-manager legibility).

## 7.1 Simulation

| | SUMO ✅ | CityFlow | PTV Vissim / Aimsun | CARLA |
|---|---|---|---|---|
| License | EPL, free | Apache, free | Commercial $$ | MIT, free |
| OSM import | Excellent (netconvert) | Weak | Good | Limited maps |
| Emissions/fuel | HBEFA model built in | ❌ | Yes | ❌ |
| Public transport | Yes (+GTFS tooling) | ❌ | Yes | ❌ |
| RL control API | TraCI + **libsumo** | Native C++ API (fast) | COM (Windows) | Python |
| Ecosystem | Standard in the field | Stagnant since ~2021 | Closed | AV-perception niche |

**SUMO 1.27 + libsumo.** CityFlow is faster per step but loses emissions, PT, OSM quality,
and community — three of our metrics die with it. libsumo (in-process bindings) removes the
TraCI socket overhead that would otherwise dominate training; the cost (one SUMO per
process) matches the Ray worker model exactly.

## 7.2 RL training

| | From-scratch MAPPO + Ray Core ✅ | RLlib | Stable Baselines3 | CleanRL as-is |
|---|---|---|---|---|
| Multi-agent CTDE | We control it | Native but API churn, heavy | Single-agent only | Single-agent recipes |
| Debuggability | Full | Notoriously hard | Good | Excellent |
| Learning/interview value | Maximum | "I wrote a config" | Medium | High |
| Risk | Bugs are ours | Version traps, opaque failures | MARL hacks needed | Needs MARL extension |

Decision reaffirmed from the design review: **write MAPPO ourselves** (CleanRL's PPO as the
style reference), use **Ray Core** only for parallel rollout workers (also the honest resume
line: "distributed rollout collection with Ray"). SB3 appears once — week-3 ramp-up, PPO on
a single junction, to build intuition against a known-good implementation before writing our
own.

## 7.3 Agent orchestration (cognitive plane)

| | LangGraph ✅ | CrewAI | AutoGen | Bare asyncio + provider SDKs |
|---|---|---|---|---|
| Control-flow model | Explicit graph/state machine | Role-based crews | Conversational | DIY |
| Durability | Checkpointed durable execution (1.2) | Limited | Limited | DIY |
| Structured output/tools | First-class | OK | OK | DIY |
| Fit for incident state machines | Exact fit | Poor (chat-shaped) | Poor | Possible but weeks of plumbing |

**LangGraph 1.2** — the Supervisor *is* a state machine (CLASSIFY→VERIFY→APPROVE→MONITOR);
LangGraph's checkpointing gives incident-survival-across-restart for free. CrewAI cut in the
design review (two orchestrators = keyword stuffing).

## 7.4 LLM providers

Chain: **Groq free tier** (llama-3.1-8b-instant, 14.4K req/day / 500K tok/day) → **Gemini 3
Flash free tier** (1,500 req/day) → **Ollama** local (correctness fallback, CPU-slow) →
**deterministic mapper** (no LLM). Client-side token buckets sized to free limits; provider
adapter interface so a paid key later is a config change. Anthropic/OpenAI paid APIs:
architecture-compatible, excluded only by the $0 constraint (stated in-doc — reviewers
should see the constraint, not guess).

## 7.5 Event bus

| | Redpanda ✅ | Apache Kafka | RabbitMQ | Redis Streams | NATS JetStream |
|---|---|---|---|---|---|
| Kafka API compat | 100% | native | ❌ | ❌ | ❌ |
| Local footprint | 1 binary, ~400 MB RAM | JVM + (ZK/KRaft), heavy | Light | Trivial | Light |
| Replay/event sourcing | ✅ | ✅ | Weak | Partial | ✅ |
| Resume recognition | Reads as "Kafka" | Highest | Medium | Low | Low |

**Redpanda locally; Kafka-the-protocol everywhere in code and docs.** Replay requirement
kills RabbitMQ; laptop footprint kills Kafka-proper; Redis Streams would conflate the bus
with the hot-state store (separate failure domains deliberately).

## 7.6 Datastores

Chosen in Part 3: **Redis** (hot state, locks, dedup) + **PostgreSQL/TimescaleDB**
(telemetry, audit, incidents). Rejected: InfluxDB (second query language, second engine for
a solo maintainer), ClickHouse (right at 100× the scale, operational overkill here),
MongoDB (nothing document-shaped that Postgres JSONB doesn't cover).

## 7.7 Services, CV, forecasting, frontend

- **FastAPI + Pydantic v2** — schema validation shared between API models and bus schemas;
  async fits consumers. (Flask/Django rejected: no async-first, no typed validation story.)
- **PyTorch** for all trained models (one framework; team-of-one rule). **Ultralytics YOLO**
  for detection (fine-tuning economics unbeatable); **ByteTrack** for tracking; **ONNX
  Runtime** for the INT8 edge profile (vendor-neutral, runs on any laptop — TensorRT noted
  as the NVIDIA-hardware path).
- **Forecasting:** own Graph WaveNet implementation in PyTorch (~300 lines; PyG dependency
  avoided — its version matrix is a known time sink and we need only one conv type).
- **Frontend: React + TypeScript + Vite + MapLibre GL + deck.gl** (vector map + WebGL data
  layers, free tiles), Zustand for state, ECharts for charts, WebSocket bridge from bus →
  UI. (Streamlit rejected for the flagship UI — reads student-grade for a "production"
  claim — but used internally for quick experiment views during development.)

## 7.8 MLOps & platform

| Concern | Choice | Rejected | Why |
|---|---|---|---|
| Experiment tracking + registry | MLflow | W&B (SaaS limits, vendor tie) | Self-hosted, registry stages drive deployment (§4.6) |
| Data/scenario versioning | DVC (Drive remote) | Git-LFS, lakeFS | Fits Colab/Drive workflow; lakeFS overkill |
| Containers | Docker + compose (primary) | — | The dev/demo environment |
| Orchestration | Helm chart on kind, week 10 | Full cloud K8s | Learning goal met without cloud spend; chart is cloud-portable by construction |
| CI/CD | GitHub Actions | Jenkins, GitLab | Free, visible on the repo, industry default |
| Monitoring | Prometheus + Grafana | Datadog (cost) | Standard OSS pair; dashboards ship in-repo |
| Logs | structlog JSON → Loki (optional) | ELK | ELK is a project unto itself |
| Python tooling | uv + ruff + mypy + pytest + pre-commit | poetry/pip, black+flake8+isort | Fast, modern, single-tool lint/format; strict typing on service boundaries |

## 7.9 The one-line stack

Python 3.12 · SUMO 1.27/libsumo · PyTorch · from-scratch MAPPO · Ray Core · Ultralytics
YOLO + ONNX INT8 · Graph WaveNet (own impl) · LangGraph 1.2 · Groq/Gemini free tier + Ollama ·
Redpanda (Kafka API) · Redis · PostgreSQL/TimescaleDB · FastAPI/Pydantic · React/TS +
MapLibre/deck.gl · MLflow · DVC · Docker/compose · Helm/kind · GitHub Actions · Prometheus/
Grafana · uv/ruff/mypy/pytest.
