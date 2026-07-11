# Part 11 — Evaluation

The evaluation is designed so that every headline claim is (a) against a credible baseline,
(b) statistically qualified, and (c) reproducible from a command. This is the part that
separates ATLAS from every "RL beats fixed-time" repo.

## 11.1 Baselines

1. **Fixed-time (Webster)** — per-junction optimal cycle/splits from calibrated demand; the
   floor every system must beat.
2. **SUMO-actuated** — gap-based green extension; representative of most real deployments.
3. **Max-pressure** — the strong decentralized adaptive baseline with stability guarantees;
   beating it is the meaningful result. (It is also our fallback — so the degradation
   experiments double as baseline runs.)

Not compared against SCATS/SCOOT directly (proprietary logic; any reimplementation would be
a strawman) — stated openly, with actuated + max-pressure as the honest proxies.

## 11.2 Traffic metrics (network-level, per run)

| Metric | Definition |
|---|---|
| Mean waiting time | Per-vehicle accumulated time at v < 0.1 m/s |
| Mean travel time / delay | Trip duration; delay = actual − free-flow |
| Throughput | Completed trips per hour |
| Queue metrics | Mean/p95 max queue per approach; spillback event count |
| Emergency response | Depot→scene time; and *network recovery time* post-preemption (return to within 10% of steady-state delay) |
| Fuel / CO₂ | SUMO HBEFA-based per-vehicle totals |
| PT efficiency | Mean bus delay vs. timetable; bus stop-to-stop time variance |
| Person-delay | Delay × occupancy factors (buses weighted properly — where PT priority proves itself) |

## 11.3 Statistical protocol

- **10 seeds per (controller × scenario)** — seeds control demand sampling, fault timing,
  and policy stochasticity; evaluation variants disjoint from training variants (§8.3).
- Report **mean ± 95% CI** (t-distribution); controller comparisons via Welch's t-test with
  Holm–Bonferroni correction across scenarios; effect sizes (Cohen's d) alongside p-values.
- One-page **claims table** in the final report: claim → number → CI → test → run ids. No
  number outside this table appears in the README.

## 11.4 Scenario suite

The 10 disruption classes (§8.3), each in ≥ 3 variants: a ~35-cell grid × 4 controllers ×
10 seeds ≈ 1,400 evaluation runs. At libsumo speeds (≥ 50× real time, 1-hour episodes) this
is a few days of background compute on Kaggle/local — budgeted in Week 11, and the harness
runs incrementally all along (nightly mini-suite in CI from Week 6).

## 11.5 Robustness & degradation experiments (the differentiator)

1. **Perception noise sweep:** twin sensor faults parameterized by the *measured* YOLO error
   model, scaled 0×–3×; plot delay vs. noise for RL and baselines. Hypothesis to test: RL
   trained with noise randomization holds a margin over max-pressure until extreme noise.
2. **Detector dropout:** 0–50% of detectors dead; success criterion: never worse than
   actuated (per NFR).
3. **Hint dropout:** corridor coordinators disabled — cost of losing coordination.
4. **Chaos tests:** kill K agent processes mid-episode (watchdog engagement time, delay
   cost); partition the bus for area X (fallback correctness); kill the preemption overlay
   mid-run (lock TTL recovery). Each is a scripted `tests/chaos/` scenario with pass
   criteria, run against compose and kind.

## 11.6 System (non-traffic) metrics

Per-hop latency p50/p99 vs. the §5.7 budget (published table); fallback engagement latency
(cycles); event loss (must be 0 outside injected faults); replay determinism check;
resource footprint of the `core` profile.

## 11.7 Model-level evaluations

| Model | Metrics |
|---|---|
| MAPPO policies | Training curves (public in MLflow), final vs. baselines, generalization to unseen variants, noise-sweep margins |
| YOLO | mAP@0.5:0.95 on UA-DETRAC val split; count MAE via virtual loops; INT8 vs. FP32: mAP delta, p50/p99 CPU latency, model size |
| Forecaster | MAE/RMSE per horizon vs. historical-average + last-value; quantile calibration; disruption-window MAE separately |
| Anomaly | Detection delay, precision/recall per incident class vs. scenario ground truth |
| LLM cognitive layer | Scripted suite of ≥ 30 seeded incidents: playbook-selection accuracy (target ≥ 80%), parameter sanity rate, grounding rate (% claims citing what-if ids), refusal correctness, deterministic-fallback parity; per-provider comparison (Groq vs. Gemini vs. Ollama) |
| XAI | Human-check rubric on 20 sampled explanations: faithful to logs? actionable? (self+peer scored — small-N, stated as such) |

## 11.8 "How do we know the system is good?" — the four-sentence answer

The controller beats fixed-time, actuated, and max-pressure on mean delay with
non-overlapping 95% CIs across a 10-class disruption suite, and degrades to no-worse-than-
actuated under 50% sensor loss. Emergency runs are ≥ 30% faster with quantified network
recovery. The fast loop meets a measured sub-second p99 sensing-to-command budget and
survives scripted process/bus failures. The cognitive layer selects correct playbooks ≥ 80%
of the time on a scripted incident suite, grounds its recommendations in counterfactual
simulations, and the whole thing reproduces from a git SHA + DVC hash + seed.
