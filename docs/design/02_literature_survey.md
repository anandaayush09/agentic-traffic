# Part 2 — Literature Survey

Scope: deployed adaptive signal systems (SCATS, SCOOT, SURTRAC), academic MARL traffic
control, intelligent transportation systems (ITS) and smart-city platforms, adaptive signal
control theory, traffic prediction, and digital twins. Each entry covers architecture,
algorithms, data, strengths, weaknesses, deployment status, and limitations. §2.9 distills the
opportunities ATLAS exploits.

---

## 2.1 SCATS (Sydney Coordinated Adaptive Traffic System)

- **Architecture.** Hierarchical and centralized: regional computers supervise groups of
  intersections; local controllers execute plans. Intersections are grouped into "subsystems"
  sharing a common cycle length; subsystems "marry" and "divorce" dynamically to form
  corridors.
- **Algorithms.** Not model-based optimization — a library of incremental heuristics. Stop-line
  detectors measure *degree of saturation* (DS); SCATS adjusts cycle length, green splits, and
  offsets in small steps between cycles to hold DS near a target. Plan selection is by voting
  among candidate plans.
- **Data.** Inductive loop detectors at stop lines; increasingly radar/video virtual loops.
- **Strengths.** Proven at massive scale (deployed in 180+ cities, tens of thousands of
  intersections, notably Sydney, Melbourne, Dublin, Shanghai); robust; tolerable degradation
  when detectors fail (falls back to plans).
- **Weaknesses.** Cycle-based adaptation (reacts over minutes); heuristics need expert tuning
  per city; corridor structure is semi-static; no anticipation (purely reactive to measured
  saturation); proprietary and expensive.
- **Deployment.** Commercial product of Transport for NSW; one of the two dominant systems
  worldwide since the 1970s–80s.
- **Limitations for our problem.** No learning; no incident reasoning; emergency preemption is
  local and disruptive; no forecasting; explanation limited to engineer-readable plan states.

## 2.2 SCOOT (Split Cycle Offset Optimisation Technique)

- **Architecture.** Centralized: a central computer receives second-by-second detector data
  from upstream (not stop-line) detectors and continuously runs an online traffic model.
- **Algorithms.** Maintains cyclic flow profiles per link from upstream detectors, predicts
  queue evolution with a platoon-dispersion model, and makes frequent *small* optimizations to
  Splits, Cycle time, and Offsets (hence the name) — a hill-climbing scheme that avoids large
  plan jumps.
- **Data.** Upstream loop detectors (~100–300 m from stop line), giving arrival profiles
  rather than just presence.
- **Strengths.** Genuine online model of traffic (not just measurement); smooth adaptation;
  well-validated delay reductions (~12% vs. good fixed-time in classic UK trials);
  deployed in 250+ towns/cities (UK, Toronto, Beijing, Santiago).
- **Weaknesses.** Central single point of failure; detector placement is critical and
  expensive; small-step optimization cannot restructure control under disruption; model
  assumptions break in oversaturation.
- **Limitations for our problem.** Same class of gaps as SCATS: no learning, no disruption
  playbooks, minimal anticipation, no operator-facing reasoning.

## 2.3 SURTRAC (Scalable Urban Traffic Control, CMU)

- **Architecture.** Fully decentralized: each intersection runs its own scheduler and
  communicates *projected outflows* to downstream neighbors — coordination emerges from
  message passing, not central optimization.
- **Algorithms.** Each intersection solves a single-machine scheduling problem over detected
  vehicle clusters (dynamic programming over an aggregated arrival representation) on a
  rolling horizon (~every second). Downstream neighbors incorporate the announced outflows
  into their own arrival predictions.
- **Data.** Video detection of approaching clusters at each intersection.
- **Strengths.** Real-time (second-scale) decisions; scalable by construction; strong pilot
  results in Pittsburgh (~25% travel-time and ~30–40% wait reductions in the East Liberty
  pilot); handles irregular grid networks.
- **Weaknesses.** Myopic per-intersection objective (schedule-based, no learned long-horizon
  value); coordination is one-directional (information flows downstream only); commercialization (Rapid
  Flow / Miovision) narrows public detail; no explicit incident layer.
- **Deployment.** Pittsburgh pilot (2012–), later commercial deployments via Rapid Flow
  Technologies.
- **Limitations for our problem.** The closest philosophical ancestor to ATLAS's control
  plane (decentralized, real-time, message-passing) — but no learning from experience, no
  robustness quantification, no cognitive layer.

## 2.4 MARL traffic signal control (academic line)

Key systems, in rough chronology:

| System | Idea | Algorithm |
|---|---|---|
| IntelliLight (KDD'18) | Deep RL for a single real-data intersection; phase-gated network | DQN variants |
| FRAP (CIKM'19) | Phase-competition invariant architecture (symmetry under flipping/rotation) | DQN |
| PressLight (KDD'19) | Reward = intersection *pressure* (from max-pressure theory) | DQN |
| CoLight (CIKM'19) | Graph attention over neighbor intersections for coordination | DQN + GAT |
| MPLight (AAAI'20) | Parameter sharing + pressure reward, scaled to 2,510 signals (Manhattan sim) | DQN (FRAP base) |
| AttendLight / Efficient-XLight / advanced lines (2020–) | Attention over lanes/phases, universal models across intersections | DQN/PPO variants |

- **Architecture.** One agent per intersection; observation = lane queues/waves/phase;
  action = phase selection or phase switch; centralized training, decentralized execution
  (CTDE) in the better works; parameter sharing across agents is the key scaling trick.
- **Datasets/simulators.** CityFlow and SUMO; Hangzhou/Jinan real-count datasets; synthetic
  grids; (LibSignal benchmarks both).
- **Strengths.** Consistent 10–30% delay improvements over fixed-time and actuated baselines
  in simulation; pressure-based rewards connect to max-pressure stability theory; parameter
  sharing makes 100+ intersections trainable.
- **Weaknesses.** Almost universally: clean observations (no sensing noise), no safety
  shield, no fallback design, no disruption scenarios beyond demand replay, no deployment
  path, evaluation often single-seed. Reproducibility across papers is poor (LibSignal was
  created to fix this).
- **Deployment.** Essentially none in production signal systems to date.
- **Limitations → ATLAS's exact opening.** ATLAS adopts the sound core (CTDE, parameter
  sharing, pressure-informed rewards) and supplies what the literature omits: calibrated
  sensor noise, shields and degradation ladders, disruption suites, seeds and CIs, and a
  production event-driven implementation.

## 2.5 Intelligent Transportation Systems (ITS) & smart-city platforms

- **Architecture.** Standards-driven layers: roadside units and detectors → field networks →
  traffic management centers (TMC) → traveler information. Standards: NTCIP (device comms),
  SAE J2735 (V2X messages: SPaT — Signal Phase and Timing, BSM — Basic Safety Message),
  ISO/ETSI C-ITS stacks. Modern platforms (e.g., NVIDIA Metropolis, Bosch, Siemens Mobility,
  open-source FIWARE) add video analytics and data buses.
- **Algorithms.** Mostly rule/threshold-based ATMS logic: incident detection (e.g.,
  California algorithm on occupancy discontinuities), ramp metering (ALINEA), traveler info.
- **Strengths.** Interoperability; hardened field equipment; TMC operational practice
  (playbooks!) that ATLAS's cognitive plane deliberately mirrors.
- **Weaknesses.** Intelligence is thin: detection thresholds, manual response plans, human
  dispatch. Data silos between signal systems, transit, and emergency services.
- **Limitations.** The playbook concept exists but is executed by humans reading screens;
  ATLAS automates playbook selection + simulation-backed verification with human approval —
  a direct upgrade of existing TMC practice, which makes the design legible to domain experts.

## 2.6 Adaptive signal control theory

- **Webster (1958):** closed-form optimal cycle/splits for isolated undersaturated
  intersections — the fixed-time baseline generator.
- **Vehicle actuation:** gap-out/max-out green extension — the second baseline.
- **Max-pressure / backpressure (Varaiya 2013, from Tassiulas–Ephremides network scheduling):**
  choose the phase maximizing Σ(upstream queue − downstream queue)×saturation flow. Provably
  maximum-stability under assumptions (infinite lane capacity, known turn ratios), fully
  decentralized, model-free, O(phases) per decision.
- **Role in ATLAS.** Max-pressure is *both* the strongest classical baseline *and* the safety
  fallback: it needs only queue estimates, no training, and no communication — precisely what
  you want when the learned layer is degraded. This dual use is a deliberate design decision.

## 2.7 Traffic prediction

- **Classical:** ARIMA/Kalman on loop series — fails on nonlinearity and spatial coupling.
- **Deep spatio-temporal line:** DCRNN (diffusion convolution + encoder-decoder GRU,
  ICLR'18), STGCN (Chebyshev graph conv + gated temporal conv), Graph WaveNet (adaptive
  adjacency + dilated temporal convolutions, IJCAI'19), ASTGCN/GMAN (attention), and
  post-2020 refinements. Benchmarks: METR-LA (207 sensors, LA highways), PEMS-BAY (325).
- **Strengths.** 15–60-min horizon accuracy well above classical baselines; graph structure
  matches road networks; Graph WaveNet-class models are compact (~300K params) and train in
  hours on a single GPU — feasible on Colab.
- **Weaknesses.** Highway benchmarks dominate (arterial data scarce); models predict flow,
  not incidents; distribution shift under disruptions is exactly where they fail.
- **ATLAS's use.** Train a Graph WaveNet-class model on ATLAS's *own* district telemetry
  (same graph as control), horizons 15/30/60 min; consume forecasts in the cognitive plane
  (anticipatory playbooks) and dashboard. METR-LA is retained only as an optional external
  benchmark to prove the implementation is honest.

## 2.8 Digital twins for mobility

- **Landscape.** SUMO (microscopic, open source, TraCI/libsumo control APIs, OSM import,
  emissions models — the academic standard); CityFlow (faster RL-oriented microsim, but
  sparse ecosystem); Aimsun/PTV Vissim (commercial, high fidelity, licensed); CARLA (3D
  perception-grade, single-vehicle scale, far too heavy for district control on our compute).
- **Twin patterns.** (1) Offline scenario lab; (2) online shadow synchronized with reality;
  (3) *what-if service* — clone current state, apply hypothetical, fast-forward, compare.
- **Strengths/weaknesses.** SUMO's calibration burden is real (demand estimation dominates
  fidelity); its emissions model (HBEFA-based) is standard enough to cite; libsumo removes
  the TraCI socket bottleneck at the cost of single-process binding — which our
  Ray-parallel-worker design accommodates naturally (one SUMO per worker process).
- **ATLAS's use.** SUMO is the environment, the evaluation harness, *and* — pattern (3) — a
  first-class counterfactual tool exposed to the LLM supervisor. Making the twin callable by
  an agent is the most novel systems element in ATLAS.

## 2.9 Opportunities ATLAS exploits

1. **Bridge SURTRAC's decentralized real-time scheduling with MARL's learning** — learned
   per-intersection policies + corridor message passing, under CTDE.
2. **Import TMC playbook practice into an agentic architecture** — LLM incident commander
   selecting/parameterizing playbooks, verified by counterfactual simulation, gated by human
   approval: no deployed or academic system does this end-to-end.
3. **Answer the robustness question the MARL literature skips** — quantify control
   degradation under perception noise measured from a real YOLO model and under
   detector/communication dropout.
4. **Treat max-pressure as both baseline and fallback** — connecting learned control to
   stability theory and to a production degradation ladder.
5. **Evaluate like a systems paper, engineer like a product** — seeds, CIs, scenario suites,
   chaos tests, latency SLOs; event sourcing, replay, monitoring, CI/CD.
