# Part 9 — Repository Design

One monorepo: `atlas-traffic`. A recruiter forms an opinion in 90 seconds; the README's
first screen carries a demo GIF, an architecture diagram, headline numbers with CIs, and
badges (CI, coverage, license). A senior engineer forms an opinion in 10 minutes; the
layout below is optimized for that reading path.

```
atlas-traffic/
├── README.md                  # GIF, architecture, headline results, quickstart (3 commands)
├── pyproject.toml             # uv-managed single Python project (src layout)
├── docker-compose.yml         # full local bring-up; profiles: core|full|demo
├── Makefile                   # make up / train / eval / test / demo — the golden paths
├── .github/workflows/         # ci.yml (lint+type+test), eval.yml (nightly mini-suite),
│                              # docker.yml (image builds), docs.yml
├── configs/                   # Hydra-style YAML: env, training, agents, scenarios, deploy
│   ├── network/  training/  scenarios/  services/  playbooks/
├── schemas/                   # versioned JSON Schemas for every bus topic (Part 5)
├── src/atlas/
│   ├── twin/                  # SUMO wrappers: env (libsumo), bus adapter, sensor fault
│   │                          # model, scenario engine, what-if service
│   ├── control/               # intersection agent runtime, safety shield, watchdog,
│   │   ├── policies/          # MAPPO actor/critic defs; max_pressure.py; fixed_time.py
│   │   ├── corridor/          # coordinators (green wave, spillback)
│   │   └── preemption/        # overlay state machine, lock protocol
│   ├── marl/                  # from-scratch MAPPO: buffers, GAE, updates, distributed
│   │                          # rollout (Ray), curriculum runner  ← reviewers read this first
│   ├── cognitive/             # LangGraph graphs, tools, playbook registry, provider
│   │                          # chain (groq/gemini/ollama/deterministic), token buckets
│   ├── perception/            # YOLO fine-tune, ByteTrack counting, error-model fit,
│   │                          # ONNX INT8 export + bench
│   ├── forecasting/           # Graph WaveNet impl, dataset builders, quantile heads
│   ├── anomaly/               # EWMA tier + LSTM-AE tier
│   ├── services/              # FastAPI apps: control_api, cognitive_api, twin_api,
│   │                          # dashboard_api (ws bridge), state_writer, sinks
│   ├── bus/                   # producer/consumer wrappers, idempotency, DLQ, tracing
│   ├── evaluation/            # metrics, baselines harness, scenario suite runner,
│   │                          # statistics (CIs, tests), chaos experiments, report gen
│   └── common/                # config loading, logging, redis, db, auth (JWT)
├── frontend/                  # React+TS+Vite: map view, approval queue, XAI panel, replay
├── notebooks/                 # colab_train_marl.ipynb, colab_train_forecast.ipynb,
│                              # kaggle_yolo_finetune.ipynb — thin wrappers calling src/
├── deploy/
│   ├── docker/                # per-service Dockerfiles (multi-stage, non-root)
│   ├── helm/atlas/            # one chart; values-kind.yaml
│   └── monitoring/            # prometheus.yml, grafana dashboards (json), alert rules
├── tests/
│   ├── unit/                  # shield invariants, max-pressure math, schema validation,
│   │                          # lock protocol, reward functions
│   ├── integration/           # bus round-trips against compose, twin adapter, what-if
│   ├── evaluation/            # smoke: 2-junction net trains + beats fixed-time
│   └── chaos/                 # kill-agent, partition-bus, stale-state scenarios
├── data/                      # DVC-tracked (pointers only in git)
└── docs/
    ├── design/                # THIS document set (00–14)
    ├── adr/                   # Architecture Decision Records (one per frozen decision)
    ├── runbooks/              # bring-up, training, incident-demo scripts
    └── report/                # final technical report + evaluation figures
```

Directory rationale (the non-obvious ones):

- **`src/atlas/marl/` separated from `control/`** — training code vs. runtime policy
  execution are different lifecycles; reviewers hunting for "did they really write MAPPO"
  find a self-contained, readable package.
- **`schemas/` at top level** — contracts are first-class citizens, not buried in a
  service; CI validates every producer/consumer against them.
- **`notebooks/` are thin** — all logic lives in `src/`, notebooks only orchestrate Colab/
  Kaggle sessions; prevents the notebook-rot that kills portfolio credibility.
- **`docs/adr/`** — each frozen decision (two-loop split, Redpanda, from-scratch MAPPO,
  deterministic preemption…) gets a one-page ADR; interviewers love them and they are
  cheap because Parts 3–7 already contain the content.
- **`tests/chaos/`** — chaos experiments as *tests*, not folklore: the fallback-ladder and
  lock-TTL claims in Parts 4–5 execute in CI-adjacent runs.
- **One Python package** (not per-service packages): a solo engineer's velocity beats
  premature service isolation; Docker images select entrypoints from the same wheel.
