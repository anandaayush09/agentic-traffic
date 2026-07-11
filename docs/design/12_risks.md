# Part 12 — Risks

Risk register (P = probability, I = impact on the 12-week outcome, H/M/L), with mitigations
and — most importantly — the pre-committed descope decision for each red risk.

## 12.1 Technical risks

| # | Risk | P | I | Mitigation | Descope trigger & action |
|---|---|---|---|---|---|
| R1 | **MARL doesn't converge at district scale** (the classic failure) | M | H | Curriculum (1→5→25 junctions); pressure reward (dense, proven); parameter sharing; SB3 sanity baseline in Week 3; hyperparameters from MAPPO/MPLight literature, not from scratch | If Week 7 district training still < max-pressure: ship corridor-scale RL (5–10 junctions) + max-pressure for the rest — the hierarchy makes this a *configuration*, and the honest writeup of why is itself senior-grade content |
| R2 | **Colab/Kaggle throughput insufficient** | M | H | libsumo (not TraCI); 5 s decision interval; 25-junction cap; parallel envs; overnight sessions with 10-min checkpointing; measure steps/hour in Week 3 and re-plan immediately | Shrink to AM-peak-only episodes; reduce eval suite variants; never silently reduce seeds — cut scope, not rigor |
| R3 | **OSM network quality poor / junction geometry broken** | M | M | Selection criteria (§8.1); Week-2 hand-audit budgeted; fallback candidate district pre-selected | Switch to the well-tagged fallback district; synthetic-grid last resort (stated openly in README) |
| R4 | **Free-tier LLM limits or provider changes break the cognitive plane** | M | M | Provider chain with client-side token buckets; semantic cache; episodic cadence (< 200 calls/demo-day); deterministic mapper as terminal fallback | Demo runs on deterministic playbooks + Ollama; the LLM comparison table shrinks — system value survives by design |
| R5 | **What-if state cloning is inaccurate** (twin fork from detector state ≠ true state) | H | M | Don't oversell: what-if forks *approximate* state (demand model + detector-implied queues); validate by comparing what-if predictions vs. actual outcomes on scripted incidents and *publish the calibration error* | Reframe tool as "scenario-conditioned forecast" — still grounded, honestly labeled |
| R6 | **Event-driven complexity eats the schedule** (12 services, one engineer) | M | H | Compose profiles; `bus/` wrapper library written once; integration tests early (Week 5); no service exists without a consumer | Merge ML services into one process (deployment detail, architecture intact on paper and in interfaces) |
| R7 | **Perception error model too noisy to be meaningful** | L | M | UA-DETRAC is well-annotated; error model is aggregate statistics (robust); condition on simple covariates only (day/night, density) | Use literature-standard noise parameters, keep measured model as appendix |
| R8 | **Windows dev friction** (SUMO/libsumo, Ray on Win) | M | M | Develop in WSL2/containers from Day 1 (compose is the dev env); Colab/Kaggle are Linux anyway | — |
| R9 | **Scope creep** (this document describes a lot of system) | H | H | The roadmap (Part 13) has explicit *cut lines* per week; the frozen-architecture rule; weekly deliverable = merged, tested, demoable | Part 13's "minimum flagship" definition is the floor: twin + MARL + shield/fallbacks + bus + dashboard + evaluation. Everything else is additive |

## 12.2 Challenged assumptions (red-team notes kept alive)

1. **"RL will beat max-pressure."** Not guaranteed — max-pressure is strong. The project's
   value survives a negative result *because* the evaluation is rigorous (a well-measured
   "RL matches max-pressure but wins under PT-priority and noise" is a better story than an
   unbenchmarked +40%).
2. **"Agents = better."** The cognitive plane must prove it beats a lookup table: that is
   exactly what the deterministic-fallback parity metric measures. If the LLM only matches
   the mapper on scripted incidents, the honest conclusion goes in the report — and the
   what-if grounding (which the mapper can't parameterize freely) is where the LLM should
   earn its keep.
3. **"The twin is realistic."** SUMO + synthetic demand is a model, not truth. All claims
   are explicitly *in-simulation*; the sim-to-real section (perception error bridge, real
   detector schemas, edge profile) is the honest boundary of what this project shows.
4. **"5-second decisions are real-time."** For signal control they are (real controllers
   operate second-scale) — but the *system* p99 < 1 s claim is what makes the architecture
   real-time, and it is measured, not asserted.

## 12.3 Non-technical risks

- **Burnout / life events:** the roadmap holds a buffer week (W12) and the minimum-flagship
  cut line; every week ends deployable.
- **Reviewer skepticism of simulation projects:** countered by baselines + statistics +
  systems rigor + the explicit sim-to-real boundary — the document repeatedly chooses
  honesty over hype because that *is* the differentiator at the target level.
- **Plagiarism suspicion (portfolio inflation era):** ADRs, commit history, training logs
  in MLflow, and the from-scratch MAPPO with its own bugs-and-fixes journal (`docs/adr/` +
  training postmortems) constitute provenance.
