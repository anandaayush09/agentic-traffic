# Part 10 — Deployment

## 10.1 Environments

| Env | Purpose | Stack |
|---|---|---|
| `dev` | Daily work | docker-compose `core` profile (twin, bus, redis, pg, one agent worker) + hot-reload services on host |
| `full` | Integration/demo | compose `full` (everything incl. Grafana, frontend) — one `make up` |
| `kind` | K8s validation | Helm chart on kind; happy path: install → run scripted demo → chaos test |
| `public demo` | Recruiters | Read-only replay: frontend + dashboard_api + Postgres snapshot on a free tier (HF Spaces / Fly.io); **zero control endpoints, no bus, no LLM keys** |
| `training` | Colab/Kaggle | No containers: `pip install atlas-traffic` from the repo + libsumo wheels; artifacts to Drive/MLflow |

## 10.2 Docker

Multi-stage builds (uv-compiled wheel → slim runtime), non-root user, pinned digests,
healthchecks on every service; one image per service family (twin, control, cognitive,
ml-services, apis) sharing the base to keep pulls small. Compose profiles gate resource
usage — `core` must run in 8 GB RAM so a mid-range laptop can develop.

## 10.3 Kubernetes (scoped honestly)

One Helm chart, one target (kind), one demo path — the review's anti-time-sink scoping:

- Deployments per service; **StatefulSet** for Redpanda + Postgres; agent workers as a
  Deployment with `intersection-shard` indexed via env (HPA documented but not demoed).
- ConfigMaps from `configs/`; Secrets for JWT signing key (and LLM keys when present);
  liveness = `/health`, readiness = bus-connected.
- The chart's `values.yaml` is written cloud-portable (storage class, ingress, resources
  parameterized) — "would deploy to EKS/GKE by changing values" is a defensible claim only
  because the chart never hardcodes kind-isms.

## 10.4 CI/CD (GitHub Actions)

- **`ci.yml`** (every PR): ruff + mypy + pytest-unit (< 5 min); schema-compat check
  (produced messages validate against `schemas/`); docker build of changed images.
- **`integration.yml`** (PR label / main): compose-up core, run integration + chaos smoke.
- **`eval.yml`** (nightly, and pre-release): 2-junction training smoke ("still beats
  fixed-time after this commit") + mini scenario suite; posts metric deltas as a commit
  comment — a *model regression gate*, the MLOps detail interviewers probe for.
- **`release.yml`** (tag): full images → GHCR, Helm package, demo-site refresh.

## 10.5 MLOps loop

```
train (Colab/Kaggle) ─► MLflow run + checkpoint ─► registry: Staging
      ─► eval.yml shadow evaluation on frozen suite (10 scenarios × seeds)
      ─► human promote to Production (one CLI/UI step, audit-logged)
      ─► control plane hot-reloads on registry stage change
      ─► rollback = stage flip (previous Production retained)
```

Same registry flow for YOLO, forecaster, and anomaly models. Drift monitors (forecast MAE,
perception confidence distribution, policy fallback-engagement rate) emit `anomaly.model_drift`
→ Supervisor → operator brief: the "monitoring closes the loop" story.

## 10.6 Monitoring, logging, alerting

Prometheus scrape on all services; the four Grafana dashboards from §3.9 shipped as JSON in
`deploy/monitoring/`. Alert rules (Alertmanager → local webhook): fallback engaged > 2 min,
bus consumer lag, what-if latency, heartbeat loss, DLQ non-empty, free-tier budget 80%.
structlog JSON logs with `trace_id` correlation; optional Loki in `full` profile.

## 10.7 Security

- **Threat model (documented in `docs/design/`):** spoofed sensor events → schema + producer
  ACLs (Redpanda) + (documented for real deployment: signed messages/mTLS); command
  injection → single-writer enforcement at twin adapter + JWT on mutating APIs + audit;
  the public demo is static-read-only (no bus, no mutating routes compiled in).
- **Secrets:** `.env` locally (git-ignored), K8s Secrets on kind; no secret ever in an
  image or notebook (Colab uses its secrets manager).
- **Supply chain:** uv lockfile, Dependabot, `pip-audit` in CI, pinned base images.
- **Ops honesty:** everything auth'd is audit-logged; anything not needed by the demo is
  not exposed by the demo.

## 10.8 Scaling narrative (documented path, not built)

25 junctions/1 worker → shard agents by junction hash (stateless, Redis-backed) → partition
bus topics (already keyed) → district-per-namespace federation; training scales by Ray
workers across machines. The bottlenecks in order: SUMO stepping (solved by
district-sharding twins), Redis (cluster), Postgres ingest (Timescale hypertable batching).
Stated ceiling of the current design: ~500 junctions per district cell — and *why* (single
Redis lock authority per cell), which is the answer to the inevitable scale interview
question.
