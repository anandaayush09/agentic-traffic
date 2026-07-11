# Part 6 — AI Pipeline

Every model in ATLAS exists to answer a specific system need; each section states the need,
the chosen approach, the rejected alternatives, and how the component is evaluated.

## 6.1 Multi-Agent Reinforcement Learning (the core)

**Need.** Per-intersection signal policies that beat classical control network-wide and
degrade gracefully.

**Formulation.** Decentralized POMDP, one agent per junction (N≈25):

- **Observation (per junction, normalized):** per-incoming-lane [queue length, occupancy,
  mean speed] + current phase one-hot + phase elapsed time + time-of-day encoding +
  corridor hint vector (§4.2) + last-3-step stack. Dimension ≈ 60–90 depending on geometry;
  padded/masked to a fixed layout so one network serves all junctions.
- **Action:** choice among the junction's legal phases (typically 4–8), evaluated every 5
  simulated seconds; "keep current phase" is always legal. Action masking encodes the safety
  shield: minimum green not yet met → only "keep"; maximum green exceeded → "keep" masked
  out; transitions always route through yellow/all-red programs at the executor, never
  chosen by the policy.
- **Reward:** negative junction *pressure* (Σ upstream queues − Σ downstream queues over
  movements, PressLight-style) + small penalties for phase switching (stability) and for
  bus-stop delay when a priority request is active (PT objective enters here), with a
  global episode-level delay term mixed in via the centralized critic. Pressure is chosen
  because it is dense, local, theoretically grounded (max-pressure stability), and
  correlates with network delay; pure travel-time rewards are too sparse/delayed for the
  compute budget.

**Algorithm: MAPPO, implemented from scratch (CleanRL-style, ~500 lines).**
- CTDE: shared-parameter actor (MLP 2×256) across all agents; centralized critic sees the
  concatenated district state during training only.
- PPO chosen over off-policy (QMIX/MADDPG): PPO is stable, forgiving to tune, and
  parameter-shared PPO/DQN is the demonstrated scaling recipe in this domain (MPLight);
  MAPPO specifically is the standard strong CTDE baseline (Yu et al. 2022, "The Surprising
  Effectiveness of MAPPO"). Off-policy sample efficiency matters less when samples come
  from a local simulator, and QMIX's monotonic value factorization adds fragility a solo,
  first-time RL engineer doesn't need.
- **Why from scratch, not RLlib:** learning value (the author is new to RL — owning the
  loop is the point), debuggability, interview depth, and avoiding RLlib's API churn. Ray
  Core (not RLlib) parallelizes rollout workers: K workers × 1 libsumo env each, actor
  weights broadcast per iteration, trajectories returned per iteration — a ~100-line Ray
  pattern.

**Training plan (Colab/Kaggle reality).**
- Throughput math: libsumo steps a 25-junction district at 50–200× real time per core;
  4 Kaggle vCPUs × parallel envs ≈ 1–2M env-steps/hour. MAPPO convergence on this scale
  ≈ 5–15M steps per curriculum stage → each stage is an overnight-to-weekend session.
- Curriculum: (1) single junction, synthetic demand → sanity + hyperparameters; (2) 1
  arterial corridor (5 junctions); (3) full district, calibrated demand; (4) robustness
  fine-tune with sensor-noise/hint-dropout domain randomization.
- Every run: config-hashed, MLflow-tracked, checkpoint every 10 min to Drive/Kaggle
  datasets, fully resumable (session death is normal, not exceptional).

**Evaluation.** Vs. fixed-time (Webster), SUMO-actuated, and max-pressure on the 10-scenario
suite × 10 seeds with 95% CIs; plus degradation curves vs. detector dropout %, sensor noise
level, and hint availability (Part 11).

## 6.2 Optimization algorithms (classical layer)

- **Max-pressure** — fallback tier 1 and the strongest baseline (§2.6). No training, no
  comms, O(phases) per decision.
- **Webster fixed-time** — fallback tier 2 and baseline; computed per junction from
  calibrated demand.
- **Green-wave bandwidth maximization** — Corridor Coordinators' offset recommendations and
  the Preemption Overlay's wave scheduling.
- **What-if search** — the Supervisor's playbook parameterization runs small grid searches
  over playbook parameters (e.g., metering rate ∈ {0.6, 0.8}) via parallel what-if calls —
  simulation-based optimization, bounded to ≤ 6 candidate runs per incident for compute
  honesty.
- **Route optimization** — reroute playbooks compute alternatives via k-shortest-paths on
  the live-weighted network graph (travel times from telemetry), applied in SUMO via
  rerouter devices on a percentage of affected flow (compliance modeling).

## 6.3 Computer Vision

**Need.** Prove the perception path (camera → detections → control) and measure the error
distribution that grounds the robustness study.

- **Model:** YOLO (Ultralytics current-gen nano/small) fine-tuned on UA-DETRAC (100 videos,
  ~140K frames, 1.21M boxes; available via Kaggle). Classes: car, bus, van, others.
  Transfer from COCO; standard augmentations; ~2–4 GPU-hours on Kaggle — deliberately
  modest, this is not a detection research project.
- **Deliverables:** (1) `vision.detections` producer over recorded feeds; (2) virtual-loop
  counting (line-crossing on tracked boxes — ByteTrack) so cameras emit the same
  count/occupancy semantics as loops; (3) **error model**: per-condition (day/night,
  occlusion density) miss rate, FP rate, count MAE — fitted as the twin's sensor-noise
  parameters; (4) edge profile: ONNX export + INT8 quantization, published
  mAP/latency/size table (CPU: full vs. INT8) — the Edge AI claim, made concrete without
  owning a Jetson.
- **Rejected:** training a detector on SUMO-rendered frames (SUMO's rendering is too crude
  to be honest); CARLA photorealistic pipeline (compute far beyond budget).

## 6.4 Time-series forecasting (spatio-temporal GNN)

**Need.** 15/30/60-min link-state forecasts for anticipatory playbooks and the dashboard.

- **Model:** Graph WaveNet-class ST-GNN (graph convolutions with learned adaptive adjacency
  + dilated temporal convolutions), ~300K params, quantile heads (p10/p50/p90) for honest
  uncertainty. Chosen over DCRNN (slower RNN training), pure attention models (data-hungry),
  and LSTM-per-link (ignores spatial coupling — and the graph structure argument is the
  justified GNN use in this project).
- **Data:** ATLAS's own simulated telemetry — 5-min link aggregates from weeks of varied
  scenario runs (the data-generation loop in Part 8) — so the forecast graph *is* the
  control graph. METR-LA as optional external validation of the implementation.
- **Evaluation:** MAE/RMSE per horizon vs. historical-average and last-value baselines;
  calibration of quantiles; degradation under disruption scenarios (where it matters).

## 6.5 Anomaly detection

Two tiers (design rationale in §4.8): EWMA/z-score (explainable, zero-training, catches
sensor faults and gross incidents) + LSTM-autoencoder on 15-min multivariate windows per
junction (reconstruction error → incident signatures), trained on incident-free telemetry.
Evaluated as detection delay + precision/recall against scenario-engine ground truth
(we *know* when the accident was injected — a luxury of the twin that becomes the labeled
dataset).

## 6.6 LLM reasoning (cognitive plane)

Models: Groq `llama-3.1-8b-instant` (14.4K req/day free) primary; Gemini 3 Flash (1,500
req/day) secondary; Ollama (llama3.x 8B, CPU) offline; deterministic mapper terminal.
Techniques that matter here: strict JSON-schema outputs (function-calling style), few-shot
playbook exemplars in a cached system prompt (prompt caching extends free tiers), semantic
incident-signature cache, and graph-enforced tool grounding (§4.4). Evaluated on the
scripted incident suite: playbook-selection accuracy, parameterization sanity, grounding
rate (% of claims backed by what-if ids), and refusal correctness (Part 11).

## 6.7 Explainability (XAI)

Three mechanisms, cheapest-first: (1) **decision logs** — every command stores its input
snapshot, masked actions, and shield delta (most "why" questions are answered by showing
the queues); (2) **counterfactual probes** — the Analyst perturbs logged observations and
reports policy sensitivity ("halve the northbound queue → policy keeps phase 4");
(3) **LLM narration** strictly over (1)+(2). SHAP/attention visualization rejected: for a
60-dim control policy, counterfactuals are more faithful and more legible to operators.

## 6.8 Pipeline integration map

```
UA-DETRAC ─► YOLO fine-tune ─► error model ─┐
                                            ▼
OSM ─► network ─► scenario engine ─► SUMO twin ─► telemetry ─► forecasting GNN
                        │                │  ▲                       │
                        │                │  └── what-if calls ◄── Supervisor (LLM)
                        ▼                ▼                          ▲
                 disruption suite   MAPPO training ─► policies      │
                        │                │                          │
                        └────────────► evaluation ◄── anomaly detection
```
