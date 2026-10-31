# Manhattan-scale ATLAS: progress and handoff

Goal (agreed): ATLAS covering **all of Manhattan** (live map + ATLAS signal control) on the development
laptop (i5-1135G7, 4 cores, **7.7 GB RAM**, no NVIDIA GPU), worked until the session ends. This file is
updated after every step: what is **done and verified**, what was **measured**, and the exact command to
**continue**. Anything not listed as done was not started or not finished.

## Status

| # | Step | State |
|---|---|---|
| 1 | Stage 0: download, build, audit Manhattan | **Done, measured** |
| 2 | Core enablers: parse-once signal index, teleport/scale settings, wider control dimensions | **Done, verified** (892 tests pass; index equals the per-junction functions for all 22 Midtown signals) |
| 3 | Headless Manhattan benchmark (demand, capacity, speed, RAM) | **Done, measured** (below) |
| 4 | Scalable live map | **Done, verified in the browser** (below) |
| 5 | What-if tiles and the Manhattan evaluation | **Done, measured** (below) |

## Stage 0 results (measured, not estimated)

Source: Overpass, relation 8398124 (Manhattan), fetched 2026-10-30. Provenance is in
`data/networks/manhattan/SOURCE.json` (query, date, bytes, sha256). Drivable roads only; `service` roads
(driveways, parking aisles) excluded on purpose.

| Quantity | Manhattan | Midtown district (today) |
|---|---|---|
| OSM extract | 12.4 MB, 40,159 nodes, 9,283 ways | 115 KB |
| OSM `traffic_signals` nodes | 3,868 (several per intersection) | 42 |
| Built network (`net.net.xml`) | 39.7 MB, built by netconvert in **12 s** | 0.2 MB |
| Roads / lanes / lane-km | **11,020 / 20,233 / 1,828** | 66 / 155 / 16.5 |
| Signalized junctions | **2,670, all controllable by ATLAS** (0 skipped) | 22 |
| Signal program type | all `static` | `static` |

**The control stack's fixed sizes do not fit Manhattan** (the audit is the evidence):

- **Incoming lanes per signal: up to 26** (Midtown: 10). 59 signals exceed 10 lanes. The policy
  observation is padded to `max_lanes`, so it widens from 44 to 4x26+7+2 = 113 inputs.
- **Green phases per signal: up to 7** (Midtown: exactly 2). 377 signals (14%) have more than 2.
- Consequence: `max_lanes=10, max_green_phases=2` (hard-coded in `ControlledTwin`) would make the first big
  avenue junction raise. And a Midtown-trained RL policy cannot act here (`check_policy_fits` refuses it,
  clearly); RL for Manhattan means retraining with the wider shape (needs a GPU, e.g. Kaggle).

Reproduce:

```powershell
python scripts/fetch_city_osm.py --city manhattan      # one Overpass request, ~1 min, 12 MB
python scripts/build_network.py --city manhattan       # ~12 s
python scripts/audit_network.py --net data/networks/manhattan/net.net.xml --json data/networks/manhattan/audit.json
```

## Step 3: headless benchmark (measured)

`python scripts/benchmark_city.py --scale S --sim-seconds 600`: all 2,670 signals under ATLAS max-pressure,
libsumo, teleporting after 300 s, pm_peak demand (18,000 vehicles generated, 256 s to generate).
Setup: signal index 2.6 s (was: thousands of full parses), road index 4 s, SUMO load 7 s.

| Demand scale | Vehicles at t=600 s | Speed | Median decision step | Peak RAM | Teleports |
|---|---|---|---|---|---|
| 0.05 | 270 | 10.9x real time | 0.44 s | n/m | 0 |
| 0.5 | 2,631 | 1.6x real time | 3.6 s | 851 MB | 0 |
| 1.0 | 5,339 | 0.82x real time | 6.8 s | 854 MB | 0 |

Memory is not the limit (under 1 GB of 7.7 GB); **CPU is**. These runs shared the machine with other jobs
(tests, the live API), so treat speeds as conservative. The cold-start window is only 10 simulated minutes:
a long saturated run was not measured, so "no teleports" means "none in the first 10 minutes".

## Step 4: the live Manhattan map

`ATLAS_NETWORK=manhattan python scripts/run_dashboard_demo_api.py` (add `ATLAS_DEMO_AUTH=1` for the demo
token button; `ATLAS_DEMAND_SCALE`, default 0.25). Then the dashboard as usual.

Checked end to end in the browser: all of Manhattan renders with 2,670 junction dots; roads coloured by
speed; ~2,000 vehicles at the default scale; the overlay shows the measured speed (about 1.3-2.4x real
time while other jobs were running); a demo accident put real closures on two roads, the proposal appeared
in the approval queue, approval applied `divert_arterial` in the live twin and impact measurement started.

What changed to make that work:

- The live twin reads every road and vehicle through SUMO **subscriptions** (one call per kind), runs on
  in-process **libsumo** in city mode, and draws at most 2,500 vehicles (statistics stay exact).
- Snapshots go compact above 2,000 roads: only roads with traffic are listed. With gzip a snapshot is ~26 KB
  and the whole road geometry ~400 KB.
- Frontend: roads are uploaded once and recoloured through MapLibre feature-state (only changed roads);
  junctions are one GPU circle layer above 150 junctions; the signal panel shows the busiest 40 with search.

Known limits of the live map (not fixed in this session):

- What-if in city mode runs on a local tile (step 5), not the whole network.
- The simulation still runs in the API process (a thread), not a separate worker process. A busy tick slows
  API responses (a snapshot request took 2-3 s while other CPU-heavy jobs ran).
- The RL policy cannot act on Manhattan (observation shape differs); classical modes only.
- Vehicles are sampled evenly, not by viewport.

## Step 5: what-if tiles and the Manhattan evaluation

**What-if tiles** (`src/atlas/twin/tiles.py`). A whole-borough what-if is two full simulations per incident, so
city mode simulates a ~600 m window around the incident instead: SUMO's `netconvert --keep-edges.in-boundary`
cuts the network (ids preserved) and `cutRoutes.py` clips the demand to the traffic that really crosses it.
Tiles snap to a 300 m grid, are cached under `data/networks/<city>/tiles/`, and take ~8 s to build the first
time. The what-if runs in a separate worker process (libsumo is one simulation per process and the live twin
owns it). Any failure reports "not checked", never a guess. Measured: an accident on a busy Midtown road,
divert-traffic response: 278 -> 259 mean halted vehicles in the tile, answered through the live API in 51 s
with the live twin unaffected. On a road almost no vehicle uses, the response changes nothing (identical
numbers) -- correct, but uninformative.
Limits: a tile ignores spill-back and rerouting from outside the window; it answers "does this help locally",
not "what happens citywide". The first incident in an area pays the tile build; `TileStore.prepare()` can
build tiles ahead of time (no script yet to do it for every junction).

**Evaluation** (`scripts/run_city_evaluation.py`, report `docs/report/city_evaluation_manhattan.md`): fixed-time
vs actuated vs max-pressure, all 2,670 signals, 25% demand, 1,200 simulated seconds from an empty network,
5 seeds each (resumable; raw cells in `data/networks/manhattan/runs/evaluation_cells.jsonl`).

| Controller | Waiting (s) | Travel (s) | Throughput (veh/h) |
|---|---|---|---|
| fixed_time | 197.6 | 523.4 | 2015 |
| actuated | 107.4 | 501.0 | 2490 |
| max_pressure | 106.0 | 492.4 | 2708 |

Both adaptive controllers clearly beat fixed-time on this demand; max-pressure has the best throughput.
**Caveat that matters:** seeds only change SUMO's random choices, every run replays the same trips, so the
reported p-values and Cohen's d (up to ~150) are over-confident. This shows "consistent on this demand", not
that it generalises. Also: synthetic demand, empty-start 20-minute episodes (the network is still filling),
and no disruption scenarios were run at city scale.

## Honest limits that apply throughout

- Demand is **synthetic** (randomTrips), as it is for Midtown: plausible, not calibrated to real counts.
- All of NYC (~12,500 signals) is out of reach on this laptop for live control.
- A full-borough live simulation may run **below real time** on 8 GB of RAM. Step 3 measures this.
