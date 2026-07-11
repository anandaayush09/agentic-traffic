# Part 1 — Problem Definition

## 1.1 Problem statement

Urban signalized intersections are controlled today by timing plans that are either static
(fixed-time), locally reactive (vehicle-actuated), or centrally adaptive but built on
decades-old heuristics tuned by human traffic engineers (SCATS, SCOOT). These systems:

- optimize single intersections or pre-defined corridors, not the network;
- adapt on the scale of minutes, not seconds;
- handle disruptions (accidents, closures, weather, events) through manual operator
  intervention;
- treat emergency vehicles with crude preemption that destabilizes the surrounding network;
- provide operators with raw data, not explanations or recommended actions.

**ATLAS asks:** can a hierarchy of collaborating AI agents — learned signal-control policies at
the intersection level, algorithmic coordination at the corridor level, and LLM-based incident
reasoning at the network level — control a city district's signals better than established
baselines, while degrading gracefully under sensor failure, communication failure, and
disruption, and while explaining every decision to a human operator?

The system is developed and evaluated against a high-fidelity digital twin (SUMO) of a real
district (~25 signalized intersections extracted from OpenStreetMap), because no city will let
a portfolio project drive its signals. The architecture, however, is designed so that the twin
could in principle be swapped for real detector feeds — the interfaces (event streams of
detections, signal states, and commands) are the same ones a real deployment would use.

## 1.2 Current challenges

| Challenge | Why it is hard |
|---|---|
| Network effects | Greedily optimizing one intersection pushes queues onto neighbors; coordination is a multi-agent credit-assignment problem. |
| Non-stationarity | Demand shifts by hour, day, season, weather, and events; policies trained on one regime fail on another. |
| Partial observability | Detectors see counts and occupancy, not intent or destination; cameras fail, loops break, networks drop packets. |
| Safety constraints | Signal controllers must respect minimum green, yellow clearance, and all-red intervals — always, regardless of what any model outputs. |
| Rare events dominate | Accidents and emergency runs are rare in data but define system value; a controller graded only on average flow is graded on the easy 95%. |
| Human trust | Operators will not deploy (or keep) a black box; every automated action needs an explanation and an override path. |
| Latency and scale | Sub-second sensing-to-decision loops across hundreds of intersections exceed what a single central optimizer can serve. |

## 1.3 Existing solutions (summary — full survey in Part 2)

- **Fixed-time plans** (Webster's method): cheap, predictable, blind to demand.
- **Vehicle-actuated control:** extends green on detection; locally reactive, no coordination.
- **SCATS / SCOOT:** centralized adaptive systems deployed in hundreds of cities; robust but
  heuristic, slow-adapting, expensive to tune and license.
- **SURTRAC:** decentralized schedule-based optimization (Pittsburgh); real-time and scalable
  but per-intersection myopic, with limited learning.
- **Academic MARL (IntelliLight, PressLight, CoLight, MPLight):** strong simulated results,
  near-zero deployment; typically ignore safety constraints, sensing noise, ops tooling, and
  disruption robustness.

## 1.4 Research and engineering gaps ATLAS targets

1. **Robustness gap:** MARL papers evaluate on clean demand replay; almost none quantify
   performance under sensor noise calibrated from a *real* perception model, or under
   detector/communication dropout. ATLAS makes this a first-class experiment.
2. **Safety gap:** published policies emit raw phase choices; production controllers need
   shields and degradation ladders. ATLAS specifies and tests RL → max-pressure → fixed-time.
3. **Incident-response gap:** adaptive systems optimize steady state; disruption response is
   manual. ATLAS's cognitive plane automates triage-with-approval using counterfactual
   simulation.
4. **Explainability gap:** no deployed system explains *why* a plan changed. ATLAS logs every
   decision with state snapshots and generates operator-facing explanations grounded in
   what-if runs.
5. **Systems gap:** academic code is a script; cities need event buses, monitoring, replay,
   and audit. ATLAS is built as the production system, not the experiment.

## 1.5 Why this project matters

- **Societally:** congestion costs major economies ~1–4% of GDP in lost time and fuel;
  transport is ~15–20% of urban CO₂. Signal timing is among the cheapest levers available —
  no new roads, no new vehicles.
- **Technically:** it is a genuinely hard multi-agent problem with real-time constraints,
  partial observability, and safety requirements — the same shape as problems at Waymo,
  DeepMind (data-center cooling, network optimization), and NVIDIA (Metropolis).
- **For the author:** it demonstrates, in one coherent system, MARL from scratch, agentic LLM
  orchestration with grounded tools, computer vision with edge constraints, spatio-temporal
  forecasting, event-driven distributed systems, and a complete MLOps loop — the exact
  portfolio for ML/AI Engineer and Applied/Agentic AI Engineer roles.

## 1.6 Success metrics

Primary (network-level, vs. best baseline, averaged over 10 seeds, 95% CIs — protocol in
Part 11):

| Metric | Definition | Target vs. best classical baseline |
|---|---|---|
| Average waiting time | Mean per-vehicle time at v < 0.1 m/s | ≥ 15% reduction |
| Average travel time | Mean per-trip duration | ≥ 10% reduction |
| Throughput | Completed trips per hour | No regression |
| Emergency response time | Depot→scene time for emergency runs | ≥ 30% reduction vs. no-preemption |
| Fuel / CO₂ | SUMO HBEFA-based emission model | ≥ 8% reduction |
| PT efficiency | Mean bus delay vs. timetable | ≥ 10% reduction |

Secondary (system-level):

- Graceful degradation: under 30% detector dropout, performance no worse than the actuated
  baseline (never catastrophic).
- Latency SLO: detection→command p99 < 1 s in the fast loop (measured, published).
- Cognitive-plane quality: ≥ 80% correct playbook selection on the scripted incident suite.
- Recovery: network returns to within 10% of steady-state delay within 10 min after
  preemption or incident clearance.

## 1.7 Functional requirements

- **FR1** Control all signalized intersections in the district on a 5-simulated-second
  decision cadence via learned policies with safety shielding.
- **FR2** Detect and respond to: rush-hour surges, accidents, road closures, construction
  zones, emergency vehicle runs, VIP movements, weather-reduced capacity, public-event
  demand spikes, sensor failures, and communication failures (10 scenario classes).
- **FR3** Provide deterministic emergency-vehicle green-wave preemption with measured
  post-preemption recovery.
- **FR4** Forecast link-level flow/occupancy at 15/30/60-minute horizons and surface
  forecasts to the supervisor and dashboard.
- **FR5** Detect anomalies (incidents, sensor faults) from detector streams and route them
  as events to the cognitive plane.
- **FR6** Cognitive plane: triage incidents, select and parameterize playbooks, run
  counterfactual what-if simulations, and request operator approval for disruptive actions.
- **FR7** Human-in-the-loop: operators can approve/reject playbooks, override any
  intersection to any fallback mode, and see explanations for every automated decision.
- **FR8** Replay: any run is fully reproducible from logged events (event sourcing) and
  viewable in the dashboard.
- **FR9** Perception service converts camera frames to vehicle detections/counts; its error
  profile is measured and injectable into the twin's sensor model.
- **FR10** All models tracked in MLflow; datasets and scenarios versioned in DVC; one-command
  local bring-up via docker-compose.

## 1.8 Non-functional requirements

| NFR | Requirement |
|---|---|
| Latency | Fast loop detection→command p99 < 1 s; cognitive loop action < 60 s |
| Availability | Control plane survives failure of any single agent process; watchdog engages fallback within 2 decision cycles |
| Degradation | Explicit ladder RL → max-pressure → fixed-time; no state where signals are uncontrolled |
| Reproducibility | Every experiment reproducible from (git SHA, DVC data hash, config, seed) |
| Observability | Per-hop latency histograms, agent decisions, and rewards exported to Prometheus; Grafana dashboards; structured JSON logs |
| Security | JWT on all mutating APIs; schema validation on all bus messages; public demo strictly read-only; threat model documented |
| Scalability | Architecture scales horizontally by partitioning intersections across agent workers; documented path from 25 → 500 intersections |
| Cost | Runs on a developer laptop (16 GB) via compose; training within Colab/Kaggle free/Pro quotas; LLM usage within free tiers |
| Auditability | Every control decision logged with input snapshot, chosen action, shield modifications, and policy version |
