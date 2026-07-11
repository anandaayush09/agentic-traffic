# Part 13 — 12-Week Development Roadmap

Budget: ~250 hrs (20+ hrs/week). Each week: learning objectives → implementation → testing →
documentation → commits → deliverable → resume bullet. RL and distributed-systems ramp-ups
are front-loaded per the user's stated experience gap. Every week ends with something merged
and demoable — no "integration week" at the end.

**Minimum flagship (the floor, never cut):** digital twin + scenario engine, from-scratch
MARL beating fixed-time with CIs, safety shield + fallback ladder, event bus with 3+ real
services, operations dashboard, and a rigorous evaluation report. Everything past that
(cognitive plane depth, forecasting, CV edge profile, Helm/kind, chaos tests) is additive
and cuttable per Part 12's triggers, in reverse build order.

---

### Week 1 — Foundations & the digital twin skeleton
- **Learn:** SUMO/netconvert/TraCI vs. libsumo; docker-compose; repo hygiene (uv, ruff,
  mypy, pre-commit, GitHub Actions skeleton).
- **Build:** repo scaffold (Part 9 layout); OSM district selected + netconvert pipeline;
  minimal libsumo env wrapper that steps and reads state; `docker-compose` `core` stub
  (compose file + healthchecks, services empty).
- **Test:** unit tests for the env wrapper (step, reset, observation shape).
- **Deliverable:** `make up` boots an empty compose stack; a script runs 1000 sim-steps and
  logs junction states.
- **Resume bullet:** "Built a SUMO/libsumo-based microscopic digital twin of a 25-intersection
  real-world district from OpenStreetMap data."

### Week 2 — Network audit, demand generation, scenario engine v1
- **Learn:** OD matrix generation, duarouter, traffic demand calibration basics.
- **Build:** hand-audit every junction; `randomTrips`/`od2trips` demand pipeline with
  time-of-day profiles; scenario YAML schema + loader for 3 of the 10 disruption classes
  (closure, accident, rush-hour).
- **Test:** plausibility-invariant checks (v/c ratios, LOS distribution) as automated tests.
- **Deliverable:** a calibrated baseline demand set + 3 working scenarios, DVC-tracked.
- **Resume bullet:** "Designed a declarative, versioned traffic-scenario engine supporting
  accidents, closures, and demand surges."

### Week 3 — RL ramp-up + safety shield + classical baselines
- **Learn:** RL fundamentals via a single-junction Gym-style env + Stable Baselines3 PPO
  (explicit ramp-up given stated inexperience); action masking concepts.
- **Build:** single-junction env wrapper (Gymnasium API) around the twin; SB3 PPO baseline;
  safety shield module (min/max green, yellow enforcement) as a standalone, unit-tested
  library used by *both* the SB3 baseline and later MAPPO; max-pressure + Webster fixed-time
  implementations (also standalone, tested against hand-computed cases).
- **Test:** shield invariant tests (property-based: no shield output ever violates timing
  constraints); max-pressure unit tests.
- **Deliverable:** SB3 single-junction agent beats fixed-time on a toy network; shield and
  classical controllers merged as reusable libraries.
- **Resume bullet:** "Implemented decentralized max-pressure and Webster fixed-time traffic
  control, plus a formally-tested signal safety shield enforcing timing invariants."

### Week 4 — Multi-agent env + Ray rollout workers
- **Learn:** CTDE, parameter sharing, PettingZoo-style multi-agent API, Ray Core basics
  (this week is the distributed-systems ramp-up).
- **Build:** multi-agent env wrapper (all 25 junctions, shared observation/action spec);
  Ray Core rollout workers (K parallel libsumo envs → trajectory batches); throughput
  benchmark on Kaggle (steps/hour) → **go/no-go checkpoint against R2**.
- **Test:** env contract tests (obs/action shapes across all junctions); Ray worker
  determinism-with-seed test.
- **Deliverable:** measured steps/hour on Kaggle; multi-agent env passing contract tests.
- **Resume bullet:** "Built a distributed rollout collection pipeline with Ray Core,
  parallelizing traffic simulation across worker processes."

### Week 5 — From-scratch MAPPO, event bus skeleton
- **Learn:** GAE, PPO clipping, centralized critic design (MAPPO paper).
- **Build:** MAPPO trainer (CleanRL-style) with action masking wired to the shield; MLflow
  run tracking; Redpanda compose service + `bus/` producer/consumer wrapper library +
  first two real schemas (`sensor.detections`, `control.commands`) with CI schema-validation.
- **Test:** PPO update unit tests (loss shapes, clipping behavior on synthetic batches); bus
  round-trip integration test.
- **Deliverable:** MAPPO trains on the corridor curriculum stage (5 junctions), tracked in
  MLflow; twin publishes/consumes real bus events end-to-end for one corridor.
- **Resume bullet:** "Implemented Multi-Agent PPO (MAPPO) from scratch in PyTorch with
  centralized-training-decentralized-execution and action masking."

### Week 6 — District-scale training + control-plane services + CI eval gate
- **Learn:** curriculum learning practice; nothing new infra-wise — consolidation week.
- **Build:** curriculum stage 3 (full 25-junction district) training launched (long-running,
  overnight/weekend on Kaggle, checkpoint-resumable); Intersection Agent + watchdog as real
  FastAPI/async services on the bus; `eval.yml` nightly mini-suite (2-junction smoke +
  district vs. fixed-time) as a CI regression gate.
- **Test:** watchdog failover test (kill agent process, assert fallback engages within 2
  cycles) — first chaos test.
- **Deliverable:** district-scale training running; control plane live on compose beating
  fixed-time on a smoke scenario; CI blocks regressions.
- **Resume bullet:** "Deployed a fault-tolerant multi-agent control plane with automatic
  degradation (RL → max-pressure → fixed-time) and a CI-enforced performance regression
  gate."

### Week 7 — Corridor coordination + emergency preemption
- **Learn:** green-wave bandwidth maximization; lock-based coordination patterns.
- **Build:** Corridor Coordinator service (offset hints, spillback alerts); hint features
  wired into agent observations (retrain/fine-tune with hint-dropout randomization);
  Preemption Overlay state machine + Redis lock protocol; emergency scenario class.
- **Test:** lock-TTL recovery chaos test (kill overlay mid-run); starvation-bound test for
  conflicting emergency routes.
- **Deliverable:** measured emergency response time improvement + recovery time; corridor
  coordination A/B (with/without hints) in the eval harness.
- **Go/no-go on R1:** compare district MARL vs. max-pressure — trigger the R1 descope if
  needed before investing further training time.
- **Resume bullet:** "Designed a deterministic emergency-vehicle preemption system with
  TTL-based distributed locking and measured post-preemption network recovery."

### Week 8 — Perception service + forecasting service
- **Learn:** YOLO fine-tuning workflow; ONNX quantization; Graph WaveNet architecture.
- **Build:** YOLO fine-tune on UA-DETRAC (Kaggle GPU) + ByteTrack counting + error-model fit;
  ONNX INT8 export + latency/mAP bench; Graph WaveNet forecaster trained on accumulated
  telemetry corpus; both wired onto the bus (`vision.detections`, `forecast.link_state`).
- **Test:** perception service integration test (recorded clip → detections on bus); forecaster
  MAE regression test vs. baselines.
- **Deliverable:** perception error model feeding the twin's sensor fault model (the
  robustness bridge, R7 addressed); forecasts visible on the dashboard (basic version).
- **Resume bullet:** "Trained and quantized a YOLO vehicle-detection model with a measured
  edge-deployment latency/accuracy tradeoff, and a graph neural network for multi-horizon
  traffic forecasting."

### Week 9 — Cognitive plane (LangGraph) + anomaly detection
- **Learn:** LangGraph durable graphs, tool-calling patterns, provider-chain design.
- **Build:** Supervisor graph (CLASSIFY→VERIFY→APPROVE→MONITOR) with the what-if tool
  against the twin API; playbook registry (5–8 playbooks); provider chain (Groq→Gemini→
  Ollama→deterministic) with token buckets; Anomaly Detection service (EWMA + LSTM-AE);
  Analyst/XAI agent v1.
- **Test:** scripted incident suite v1 (10 incidents) scoring playbook accuracy + grounding
  rate; provider-fallback test (simulate Groq exhaustion → falls through chain correctly).
- **Deliverable:** an injected accident triggers detection → LLM triage → what-if
  verification → simulated operator approval → applied playbook, end-to-end, on video.
- **Resume bullet:** "Built an LLM-orchestrated incident-response agent (LangGraph) that
  grounds every recommendation in counterfactual digital-twin simulations, with a
  multi-provider fallback chain guaranteeing operation at zero API cost."

### Week 10 — Frontend, K8s/Helm, security hardening
- **Learn:** Helm chart authoring; kind; JWT auth patterns.
- **Build:** React+MapLibre dashboard (map, approval queue, XAI panel, replay scrubber);
  JWT auth on mutating APIs; Helm chart + kind deployment + scripted demo running on kind;
  threat-model doc; public read-only demo deployed (free tier).
- **Test:** kind smoke test (install → scripted demo scenario → chaos test) in CI.
- **Deliverable:** live clickable dashboard; system running on Kubernetes (kind); public
  demo link.
- **Resume bullet:** "Deployed the full stack to Kubernetes via a portable Helm chart with
  CI-validated smoke and chaos tests; shipped a public read-only operations dashboard."

### Week 11 — Full evaluation suite + statistical report
- **Learn:** experiment design at scale (seeds, CIs, multiple-comparison correction).
- **Build:** run the full scenario suite (~1,400 runs) across all 4 controllers; robustness
  sweeps (perception noise, detector dropout, hint dropout); remaining chaos tests; generate
  the statistical report (claims table, figures) programmatically from MLflow + eval DB.
- **Test:** report-generation reproducibility (rerun → identical claims table given same
  run ids).
- **Deliverable:** `docs/report/` — the full evaluation report with every claim traceable.
- **Resume bullet:** "Ran a statistically rigorous evaluation (10 seeds, 95% CIs,
  Holm–Bonferroni-corrected significance tests) across a 10-class disruption suite comparing
  learned control against three classical baselines."

### Week 12 — Polish, documentation, buffer
- **Build:** README pass (GIF, architecture diagram, headline numbers); ADRs written up
  from Parts 3–10; demo video; any R1/R2/R4-triggered descopes finalized and documented;
  fix whatever Weeks 1–11 left broken (this week is explicitly the buffer — see Part 12).
- **Test:** full CI green; kind chaos suite green; public demo smoke-checked.
- **Deliverable:** the finished, documented, publicly linkable flagship repository.
- **Resume bullet:** "Shipped ATLAS: an end-to-end agentic traffic-management platform —
  multi-agent RL, LLM-based incident response, computer vision, forecasting, and a
  production event-driven architecture — with a statistically validated evaluation against
  industry-standard baselines."

---

## Hour budget sanity check

12 weeks × ~20 hrs ≈ 240 hrs, matching the ~250 hr estimate with slack for the R1–R9 risk
buffers baked into Weeks 6–7 and the explicit Week-12 buffer. Long-running training (Weeks
5–6, 8) overlaps with service-building hours — Colab/Kaggle sessions run unattended while
implementation continues, which is why the schedule is aggressive but not unrealistic.
