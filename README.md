# ATLAS — Agentic Traffic Learning & Adaptive Signaling

> An end-to-end agentic-AI traffic-management platform: multi-agent reinforcement learning
> for real-time signal control, an LLM-based cognitive layer for incident response, computer
> vision, spatio-temporal forecasting, and a production event-driven architecture — with a
> statistically validated evaluation against industry-standard baselines.

**Status: all 12 planned weeks built.** The system runs end to end against a real 22-junction
Midtown Manhattan SUMO network. Every claim in this README is backed by a real, reproducible
run — none are aspirational. Section [What isn't done](#what-isnt-done-said-plainly) says
exactly where the honest boundaries are; nothing here papers over them.

The full frozen system design — problem definition, architecture, agent specifications,
communication protocols, AI pipeline, tech stack, evaluation protocol, and the 12-week
roadmap — lives in [`docs/design/`](docs/design/00_executive_summary.md). The rationale
behind each major architectural choice, and what actually happened when a choice met reality,
is in [`docs/adr/`](docs/adr/README.md) — 13 one-page decision records.

## Architecture

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
                         Redis (hot state) · PostgreSQL+TimescaleDB · MLflow · DVC
```

Five planes around one event backbone, with exactly two documented exceptions to "bus-only"
communication: fast-loop Redis reads, and the cognitive plane's request/response what-if call
into the twin. Full rationale for every box in this diagram: [`docs/design/03_system_architecture.md`](docs/design/03_system_architecture.md)
and [`docs/adr/`](docs/adr/README.md).

## Headline results (real, reproducible, statistically tested)

From [`docs/report/evaluation_pilot_report.md`](docs/report/evaluation_pilot_report.md) — 150
real SUMO runs (3 controllers × 5 conditions × 10 seeds × 1800 simulated seconds) against the
real Manhattan corridor, Welch's t-test + Holm-Bonferroni correction (`scipy`-backed,
[`src/atlas/evaluation/statistics.py`](src/atlas/evaluation/statistics.py)):

| Condition | Actuated vs. fixed-time | p-value | Cohen's d |
|---|---|---|---|
| No disruption | 151.3s vs. 173.8s mean wait (**−13.0%**) | 0.0002 | 2.13 |
| Accident | 150.1s vs. 177.2s (**−15.3%**) | 0.0003 | 2.05 |
| Road closure | 145.5s vs. 167.3s (**−13.0%**) | 0.0001 | 2.26 |
| Construction | 151.9s vs. 174.2s (**−12.8%**) | 0.0002 | 2.15 |
| Weather derate | 151.2s vs. 173.7s (**−13.0%**) | 0.0002 | 2.14 |

Actuated control beats fixed-time significantly in every condition tested. Max-pressure is
directionally ahead of fixed-time in every condition (Cohen's d 0.58–0.80) but does not reach
significance at this pilot's sample size — reported exactly as measured, not massaged. A
first 5-seed pilot run showed large but *non-significant* effects, which is the actual
statistical-power argument for the design's 10-seed spec, not a bug — re-running at 10 seeds
produced the significant results above.

Other measured, reproducible results:
- **Detector-dropout robustness** (`scripts/run_robustness_sweep.py`): the NFR "actuated
  never significantly worse than max-pressure" holds (proper Welch's-t + Holm-Bonferroni
  comparison, not a raw-mean check — an earlier raw-mean version of this sweep falsely
  reported a failure from pure noise) at 0%, 25%, and 50% simulated detector dropout.
- **Emergency preemption** (`scripts/eval_emergency_preemption.py`): response time drops from
  40.0s to 25.0s (**37.5%**) with the Preemption Overlay engaged, reproduced identically
  across two seeds, on a real 5-junction corridor.
- **Training throughput** (Week 4, `backend=traci`, a documented pessimistic lower bound on
  this Windows dev machine): 68,392 steps/hour on 1 Ray worker → 152,571 steps/hour on 8
  (2.23× speedup).
- **Chaos resilience** (`tests/chaos/`): agent-process death, preemption-lock TTL recovery,
  and a bus partition are all scripted, passing chaos tests — the bus-partition one caught
  and fixed a real bug (`IntersectionAgentService` used to crash its whole control loop on a
  failed bus publish; see [ADR-0002](docs/adr/0002-event-driven-architecture-redpanda-bus.md)).

## What isn't done, said plainly

Two things, both requiring resources this development environment didn't have — not silently
assumed complete, not worked around, just genuinely blocked:

1. **No MAPPO checkpoint has been trained to real convergence.** (The path to close this is now
   built and smoke-tested end to end -- time-boxed resumable training, a Kaggle bootstrap, loading
   a checkpoint into the live twin and the evaluation -- but the multi-hour GPU run itself hasn't
   happened; see [`docs/training/kaggle.md`](docs/training/kaggle.md).) The from-scratch MAPPO
   implementation ([ADR-0009](docs/adr/0009-mappo-from-scratch-not-rllib.md)) is fully built,
   unit-verified layer by layer, and proven to train correctly on short runs (beats a random
   baseline by ~66% on mean pressure). Reaching real convergence needs the multi-hour-to-
   multi-day Colab/Kaggle GPU session the design's own throughput math calls for — a resource
   this session never had. RL is therefore absent (not weakened) from every evaluation number
   above; `mode="rl"` fails loudly rather than silently evaluating an under-trained policy.
2. **No LLM result has been measured against a live key.** Real Groq and Gemini providers now
   exist with quota-aware routing and Redis-shared limits
   ([ADR-0014](docs/adr/0014-quota-aware-llm-routing-with-redis-shared-limits.md)), and are tested
   end to end against mocked HTTP and the real simulation -- but this repo has never been run
   with real API keys, so the providers' live behaviour and the LLM's playbook quality versus the
   deterministic mapper (100% accuracy / grounding / parameter sanity on the 10-incident suite,
   which validates the curated defaults, not LLM reasoning) are still unmeasured.
   `scripts/check_llm_providers.py` and `scripts/eval_scripted_incidents.py --llm` are how to
   close that gap once keys are set.

The full, precise reconciliation of these two gaps against the original risk register is
[ADR-0013](docs/adr/0013-r1-r2-r4-descope-outcomes.md). Everything else in the frozen
architecture — the digital twin, the full event-driven service mesh, the safety shield and
fallback ladder, corridor coordination, deterministic emergency preemption, perception,
forecasting, anomaly detection, the cognitive plane's what-if grounding and human-approval
gate, the dashboard, Kubernetes/Helm deployment, security hardening, and the statistical
evaluation suite above — is built and verified against real runs, not mocked.

## Try it

**Public static demo** (no backend, no keys, safe to deploy anywhere): 10 real incidents run
through the real LangGraph Supervisor against the real corridor, frozen into a static build.

```bash
cd frontend
npm install
npm run build:demo   # outputs frontend/dist -- deploy anywhere as a static site
```

**Full local system** (requires Python 3.12; SUMO ships as a Python dependency, no manual
install needed):

```bash
uv venv --python 3.12
uv sync --extra dev

# Build the real 22-junction Manhattan district from the committed OSM extract
python scripts/build_district_network.py
python scripts/generate_demand.py

# Run the test suite (483 tests; 1 integration test self-skips without a Kafka broker)
pytest

# Reproduce the headline evaluation report above
python scripts/run_evaluation_pilot.py

# Run one intersection agent's control loop + a live dashboard against it
python scripts/run_dashboard_demo_api.py   # then: cd frontend && npm install && npm run dev
```

**One command:** `make demo` (or `python scripts/run_demo.py`) starts the API and the dashboard
together, waits until both answer, and stops both on Ctrl+C. In the dashboard click
**Get demo token** (no pasting JWTs -- that button exists only under `make demo`, is local-only,
and is off whenever the API is started any other way), then try **Demo: accident**.

The dashboard is wired to a **live, closed-loop digital twin** (`run_dashboard_demo_api.py` runs the
Manhattan SUMO network in the background -- [`live_traffic.py`](src/atlas/services/live_traffic.py)
on the shared [`controlled_twin.py`](src/atlas/twin/controlled_twin.py)):

- **Map:** roads coloured by speed vs. limit, every vehicle a dot, disrupted roads haloed.
- **Signal control:** ATLAS controllers (fixed-time / actuated / max-pressure, through the safety
  shield) drive **all 22 signalized junctions** of the district; switch any one from the *Signal
  control* panel (needs an operator token). With a trained policy loaded
  (`ATLAS_RL_CHECKPOINT=<file>`), an **RL policy** option appears too -- see
  [`docs/training/kaggle.md`](docs/training/kaggle.md) for how to train one.
- **Incidents disrupt real roads:** `Demo: accident` / `Demo: closure` (or `POST /incidents` with
  `class` = accident / closure / construction / weather) inject a scenario-engine disruption.
- **Responses are simulated, then applied:** for `divert_arterial` and `meter_upstream` the what-if
  tool runs the incident on the real network with and without the playbook (same seed) and shows
  the numbers. Approving applies the playbook to the live twin (`atlas.twin.playbook_effects`);
  the dashboard then reports the *observed* queue change on the affected roads. Rerouting still
  asks a human even when it helps -- only mode changes and no-ops auto-apply.

**LLM proposals (optional).** Put `GROQ_API_KEY` and/or `GEMINI_API_KEY` in a local `.env`
(gitignored; see `.env.example`) and the Supervisor's playbook proposals come from those models
instead of the lookup table. Per incident, [`AdaptiveProviderChain`](src/atlas/cognitive/llm/router.py)
picks the provider with the most quota headroom and best health, with per-minute/day counters and
cooldowns shared through Redis (`docker compose up -d redis`; in-memory fallback if it's down). The model only
*proposes*: replies are schema-, grounding- and sanity-checked, then simulated, and disruptive
playbooks still need a human ([ADR-0014](docs/adr/0014-quota-aware-llm-routing-with-redis-shared-limits.md)).
`python scripts/check_llm_providers.py` makes one real call per configured provider;
`python scripts/eval_scripted_incidents.py --llm` scores them on the 10-incident suite.

- **Live feed and replay:** the map is pushed over a WebSocket (falling back to polling), and the
  *Replay scrubber* rewinds it through the last ~15 minutes of recorded frames (in memory; the
  design's Timescale telemetry store is still not built).
- **Durable decisions (opt-in):** `ATLAS_STATE_DB=data/state/atlas.sqlite` keeps incidents -- including
  ones awaiting approval -- across API restarts (LangGraph `SqliteSaver`). It persists the decision
  *record*, not the simulation: after a restart the road starts clean at t=0.

Honest limits: `weather_capacity_derate` and `bus_bridge` have no defensible simulation, so they stay
unchecked; `meter_upstream` is modelled as max-pressure on the upstream junctions (its numeric `rate`
is not enforced); the live "observed impact" is one non-stationary run, not a controlled test (the
what-if comparison is the controlled one). Set `ATLAS_LIVE_TRAFFIC=0` to run without the live twin.

**Kubernetes** (kind + Helm, verified against a real local cluster):
`python scripts/kind_smoke_test.py` — installs the chart, runs a scripted demo scenario, then
kills a pod and verifies the cluster self-heals.

## Docker, testing and CI/CD

Full guide, with the reasoning and a first-time setup checklist: [`docs/operations/ci-cd.md`](docs/operations/ci-cd.md).

**Architecture, in deployment terms.** Two containers and an optional third. The **API** (`atlas-api`,
Python/FastAPI) runs the supervisor graph, the live SUMO traffic simulation and the WebSocket feed. The
**dashboard** (`atlas-web`, React served by nginx) calls it from the browser. **Redis** is optional (shared LLM
rate limits; it falls back to in-memory). State is in memory unless `ATLAS_STATE_DB` points at a persistent
file. The API is deliberately a *single instance*: the simulation and incident state live in that process.

**Prerequisites:** Python 3.12 + [uv](https://docs.astral.sh/uv/), Node 22+, and Docker (for the container steps).

**Local development**
```bash
uv sync --extra dev                       # add --extra training for the MLflow-based training scripts
python scripts/build_district_network.py && python scripts/generate_demand.py   # once
make demo                                 # API + dashboard together (see "Try it" above)
```

**Tests and checks** (exactly what CI runs)
```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy      # lint + security rules + format + types
uv run pytest --cov=atlas --cov-fail-under=90                           # ~820 tests, ~94% coverage
cd frontend && npm ci && npm run lint && npm run build                  # dashboard
uvx pip-audit -r <(uv export --frozen --no-hashes --no-emit-project)    # dependency vulnerabilities
```
The external LLM providers and Redis are mocked in tests (no keys, no paid calls, no services needed).

**Docker**
```bash
docker compose --profile app up --build            # API + dashboard + Redis; open http://localhost:8080
docker compose logs cognitive_api | grep -A1 "Operator JWT"    # the local operator token
docker build -t atlas-api .                        # API image (builds the road network inside)
docker build -t atlas-web frontend                 # dashboard image
docker run -p 8000:8000 atlas-api                  # then: python scripts/smoke_test_deployment.py --base-url http://localhost:8000
```

**CI/CD pipeline** (`.github/workflows/ci.yml` + `deploy.yml`). On every push/PR, **CI** runs lint/format/types,
the test suite, the frontend build and a dependency audit; if all pass it builds both images, starts the API
container and smoke-tests it. On the default branch only, **CD** then publishes `ghcr.io/<owner>/atlas-api:sha-<commit>`,
deploys that image to **staging**, verifies the *new* version is the one answering, and, after a manual
approval, promotes the *same image* to **production**. A failing check anywhere stops everything after it.

**Required configuration** is environment variables (documented in [`.env.example`](.env.example), placeholders
only): `ATLAS_JWT_SECRET`, `ATLAS_DASHBOARD_ORIGINS` and `ATLAS_ENV` are mandatory outside development, and the API
**refuses to start** in staging/production without safe values. Deploy credentials are GitHub *Environment*
secrets (`DEPLOY_HOOK_API`, ...), never in the repo.

**Deploying:** merge to the default branch; staging deploys itself; approve production in the Actions tab.
**Rolling back:** Actions -> *Deploy* -> *Run workflow* with the previous good `sha-...` tag, which is verified by
the same smoke test. **Troubleshooting:** the guide's [table](docs/operations/ci-cd.md#troubleshooting) covers
failed jobs, container exits, CORS/URL mismatches and "NOT DEPLOYED" messages.

## Documentation map

| Doc | Contents |
|---|---|
| [`docs/design/`](docs/design/00_executive_summary.md) | The frozen design: problem, architecture, agents, comms, AI pipeline, tech stack, data, evaluation protocol, risks, 12-week roadmap |
| [`docs/adr/`](docs/adr/README.md) | 13 one-page Architecture Decision Records — why each major choice was made, and what happened when it met reality |
| [`docs/report/evaluation_pilot_report.md`](docs/report/evaluation_pilot_report.md) | The real, reproducible statistical evaluation behind the headline numbers above |
| [`docs/operations/ci-cd.md`](docs/operations/ci-cd.md) | CI/CD pipeline, Docker, environments, secrets, release, rollback, logs, troubleshooting |
| [`docs/training/kaggle.md`](docs/training/kaggle.md) | Training the MAPPO policy in the cloud and bringing it back |
| [`docs/security/threat_model.md`](docs/security/threat_model.md) | What's actually authenticated/authorized, and the honestly-named gaps |
| [`data/networks/manhattan_midtown/AUDIT.md`](data/networks/manhattan_midtown/AUDIT.md) | The real network's provenance and quality audit |

## Repository layout

Full target layout and rationale: [`docs/design/09_repository_design.md`](docs/design/09_repository_design.md).

```
agentic-traffic/
├── docs/{design,adr,report,security}/  # design docs, ADRs, evaluation report, threat model
├── configs/{network,demand,fleet,scenarios}/  # declarative YAML configs
├── schemas/                # versioned JSON Schemas for every bus topic
├── scripts/                # build/train/eval/export entry points (all real, runnable)
├── src/atlas/
│   ├── common/, twin/, control/, coordination/   # digital twin + control plane
│   ├── rl/, marl/                                # single- and multi-agent RL
│   ├── bus/, services/                            # event bus + FastAPI services
│   ├── perception/, forecasting/, anomaly/         # ML services
│   ├── cognitive/                                  # LangGraph supervisor, playbooks, XAI
│   └── evaluation/                                 # metrics, statistics, scenario runner
├── frontend/                # React + TypeScript + MapLibre operations dashboard
├── deploy/helm/atlas/       # Helm chart, verified on a real kind cluster
├── tests/{unit,integration,chaos}/
└── pyproject.toml
```

## Development

```bash
ruff check .          # lint
ruff format .          # format
mypy                    # type-check (strict)
pytest                  # full test suite (unit + integration + chaos)
```

CI (`.github/workflows/ci.yml`) runs these plus a coverage floor, the frontend build, a dependency audit
and a container smoke test on every push/PR (`make check` runs the Python half locally); a nightly workflow
(`eval.yml`) re-runs a smoke-sized evaluation as a regression gate. `make help` lists the
same commands plus `train`/`eval`/`demo`/`up`/`down` shortcuts — `up`/`down` wrap `docker
compose`, exercised for real via `scripts/kind_smoke_test.py`'s Helm/kind path rather than
`docker compose up` directly.

## License

MIT.
