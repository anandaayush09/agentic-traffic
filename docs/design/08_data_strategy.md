# Part 8 — Data Strategy

## 8.1 The map (network)

- **Source:** OpenStreetMap extract of a real district, ~25 signalized intersections
  including at least two arterials, a grid section, and bus routes.
- **Selection criteria (in order):** OSM signal-tag quality (`highway=traffic_signals`
  complete), lane-count tag coverage, presence of GTFS-mappable bus routes, personal
  relevance. Primary candidate: a Manhattan-style well-tagged grid district (excellent OSM
  + public GTFS); documented alternative: a Bengaluru district with manual signal curation
  (more meaningful locally, ~1–2 extra days of hand-fixing — the tradeoff is stated, the
  choice is a config).
- **Pipeline:** `osmWebWizard`/`netconvert` → hand-audit every junction (movements, lanes,
  phase programs sane) → canonical `network.net.xml` versioned in DVC. The hand-audit is
  budgeted (Week 2): garbage junction geometry is the #1 silent killer of signal RL results.

## 8.2 Demand (traffic generation)

- **Base demand:** time-of-day origin–destination matrices built with SUMO's
  `randomTrips`/`od2trips` + `duarouter`, shaped to realistic profiles (AM peak, midday,
  PM peak, night) with turn-ratio calibration at audited junctions.
- **Calibration target:** since no ground-truth counts exist for a synthetic district, we
  calibrate to *plausibility invariants* (arterial v/c ratios 0.5–0.9 by period,
  intersection LOS distribution, realistic queue clearance) and document them — honest
  synthetic-data methodology instead of fake "real data" claims.
- **Fleet mix:** cars, buses (GTFS-derived fixed routes + stops), trucks, emergency
  vehicles; passenger occupancy factors recorded for person-delay metrics.
- **Randomization:** every training/eval episode samples demand scale (±20%), OD noise, and
  departure jitter under a named seed — the domain-randomization axis that, with the sensor
  fault model, defines the robustness study.

## 8.3 Scenario engine (the 10 disruption classes)

Declarative YAML scenarios — versioned in DVC, loaded by the twin, announced on
`scenario.events` (ground truth for evaluation):

| # | Scenario | Twin mechanism |
|---|---|---|
| 1 | Rush-hour surge | Demand scale ×1.5–2.0 on OD subsets |
| 2 | Accident | Lane block via stopped vehicle + capacity derate, timed clearance |
| 3 | Road closure | Edge closure + rerouter activation |
| 4 | Construction | Long-duration lane drop + speed limit |
| 5 | Emergency run | Emergency vehicle insertion, depot→scene, triggers overlay |
| 6 | VIP movement | Scheduled convoy route, lower preemption class |
| 7 | Weather | Global/area speed + capacity derate (rain/fog factors from literature) |
| 8 | Public event | Localized OD spike + parking search traffic around a venue |
| 9 | Sensor failure | Fault model: dropout/freeze/bias per detector set |
| 10 | Comms failure | Bus partition / consumer pause per area |

Each has scripted variants (location, severity, duration) → the **evaluation suite** is 10
classes × variants × 10 seeds. Training sees a *disjoint* variant set from evaluation
(generalization, not memorization).

## 8.4 External datasets

| Dataset | Use | Access |
|---|---|---|
| UA-DETRAC (~140K frames, 1.21M boxes) | YOLO fine-tune + error model | Kaggle mirror |
| METR-LA / PEMS-BAY | Optional forecasting benchmark | Public |
| GTFS feed of chosen city | Bus routes/stops/timetables | Public |
| HBEFA emission factors | Built into SUMO's emission model | Bundled |

## 8.5 Sensor streams & camera feeds

Defined in Parts 3/5 (schemas, fault model). Data-strategy additions: the **telemetry
corpus** — weeks of simulated operation across scenario mixes, materialized as 5-min link
aggregates (Timescale continuous aggregates → Parquet exports, DVC-tracked) — is the
training data for forecasting and anomaly models. Target: ≥ 60 simulated days across regime
mixes (a few real-time hours at libsumo speeds), regenerable from (network hash, scenario
set, seeds) — data as code.

## 8.6 Synthetic data policy

Everything except UA-DETRAC/GTFS/METR-LA is synthetic by design. Rules: (1) every synthetic
artifact regenerable from versioned config + seed; (2) calibration assumptions documented
next to the generator; (3) the perception error model is *measured from a real model on
real video* — the one deliberate bridge between synthetic and real that makes the
robustness claims defensible.

## 8.7 Versioning & lineage

- **Git:** code + configs + schemas. **DVC:** network, scenarios, telemetry Parquet,
  UA-DETRAC subset, model-eval artifacts (Drive remote — fits the Colab workflow).
  **MLflow:** runs, params, metrics, model artifacts + registry stages.
- **Lineage rule:** every MLflow run logs `git_sha`, `dvc_hash(data)`, `config_hash`, and
  `seed`. Every published number in the final report traces to (run_id, eval suite version).
  CI fails the evaluation job if lineage fields are missing — enforced, not aspirational.
