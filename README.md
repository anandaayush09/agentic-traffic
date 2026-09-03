# ATLAS — Agentic Traffic Learning & Adaptive Signaling

> **Status: Week 3 of 12 (in progress) — RL Ramp-Up.**
> This README will grow a demo GIF, architecture diagram, and headline evaluation
> results as later milestones land. Right now it documents what actually exists.

ATLAS is a production-grade, agentic-AI traffic management system: a hierarchy of
collaborating AI agents (multi-agent reinforcement learning for real-time signal control,
an LLM-based cognitive layer for incident response, computer vision, and spatio-temporal
forecasting) managing a simulated ~25-intersection city district end to end, built as a
flagship AI-engineering portfolio project.

The full frozen system design — problem definition, literature survey, architecture,
agent specifications, communication protocols, AI pipeline, tech stack, evaluation
protocol, 12-week roadmap, and the multi-persona senior-engineer design review — lives in
[`docs/design/`](docs/design/00_executive_summary.md). **Read that first.** This README
tracks implementation status against that plan; it does not re-derive the architecture.

## What exists right now

**Week 1 — digital twin wrapper:**
- A `SumoTwinEnv` wrapper ([`src/atlas/twin/env.py`](src/atlas/twin/env.py)) around a real
  SUMO simulation (via `libsumo`) that can reset, step, and read back per-junction signal
  state and per-lane traffic state (queue length, occupancy, speed, vehicle count).
- A procedurally generated placeholder network (3×3 grid, 4 signalized junctions,
  [`scripts/generate_sandbox_network.py`](scripts/generate_sandbox_network.py)) used to
  prove the twin end-to-end.
- Shared infrastructure: YAML-driven config loading validated by Pydantic models
  ([`src/atlas/common/config.py`](src/atlas/common/config.py)), structured logging via
  `structlog` ([`src/atlas/common/logging.py`](src/atlas/common/logging.py)), and
  zero-install SUMO binary discovery
  ([`src/atlas/common/sumo_locate.py`](src/atlas/common/sumo_locate.py)).

**Week 2 (complete) — real district network + demand:**
- A real 22-signalized-junction SUMO network built from OpenStreetMap (Midtown
  Manhattan, W 42nd–48th St / 6th–9th Ave), acquired and audited per
  [`data/networks/manhattan_midtown/AUDIT.md`](data/networks/manhattan_midtown/AUDIT.md)
  — connectivity-checked (single component, 100% coverage), loads and steps cleanly
  through the same `SumoTwinEnv` from Week 1.
  ([`scripts/build_district_network.py`](scripts/build_district_network.py))
- Four time-of-day demand periods (night/midday/am_peak/pm_peak,
  [`scripts/generate_demand.py`](scripts/generate_demand.py)), validated against
  plausibility invariants rather than fabricated ground truth: vehicle-count ordering,
  a turn-direction sanity audit
  ([`src/atlas/twin/demand.py`](src/atlas/twin/demand.py)), and a congestion proxy that
  falls monotonically with demand — full results and an important calibration caveat in
  AUDIT.md.
- A scenario engine (accident/closure disruptions; rush-hour is the am_peak/pm_peak
  demand periods above) that dynamically applies/reverts lane-level effects on a running
  simulation ([`src/atlas/twin/scenario.py`](src/atlas/twin/scenario.py)) — building it
  surfaced and fixed a real robustness gap in `SumoTwinEnv` itself (a closure with no
  alternate route could crash the whole simulation; see AUDIT.md for the fix).
- A fleet mix (passenger/truck/bus/emergency vehicle types, realistic proportions) layered
  onto every demand period ([`src/atlas/twin/fleet.py`](src/atlas/twin/fleet.py)) —
  "bus" here is a vehicle type mixed into random demand, not scheduled service on real
  GTFS routes, which is explicitly deferred (see AUDIT.md).

**Week 3 (in progress) — RL ramp-up:**
- The signal safety shield ([`src/atlas/control/shield.py`](src/atlas/control/shield.py))
  — enforces minimum/maximum green time on every signal decision, independent of whatever
  produces the requested action (RL policy, classical controller, human operator). SUMO-
  agnostic and unit-tested exhaustively with property-based tests (Hypothesis), which
  caught a real boundary-condition bug during development: an invalid phase request
  arriving exactly at `max_green_s` could silently bypass the max-green override.
- Not yet done: classical controllers (max-pressure, fixed-time) and the single-junction
  Stable-Baselines3 PPO baseline — see
  [`docs/design/13_roadmap.md`](docs/design/13_roadmap.md), Week 3.

Everything else described in `docs/design/` (MARL control, event bus, cognitive plane,
perception, forecasting, dashboard, deployment) is designed but **not yet built** — see
[`docs/design/13_roadmap.md`](docs/design/13_roadmap.md) for the week-by-week plan.

## Quickstart

Requires Python 3.12. No manual SUMO installation is needed — `eclipse-sumo` bundles the
full SUMO binary distribution and is installed as a normal Python dependency.

```bash
# Install uv (https://docs.astral.sh/uv/) if you don't have it, then:
uv venv --python 3.12
uv pip install -e ".[dev]"

# Generate the Week 1 placeholder network (idempotent; regenerate with --force)
python scripts/generate_sandbox_network.py

# Run 1000 simulation steps and log junction state along the way
python scripts/run_twin_smoke_test.py

# Build the real Week 2 district network from the committed OSM extract
python scripts/build_district_network.py

# Generate fleet-mixed, time-of-day demand for it (4 periods x 4 vehicle types)
python scripts/generate_demand.py

# Run the test suite
pytest
```

## Repository layout

See [`docs/design/09_repository_design.md`](docs/design/09_repository_design.md) for the
full target layout and the rationale behind it. Current state:

```
agentic-traffic/
├── docs/design/          # frozen architecture (15 files, Parts 1-14 + exec summary)
├── configs/
│   ├── network/            # declarative YAML network/twin configs
│   ├── demand/              # time-of-day demand period recipes
│   ├── fleet/                # vehicle-type mix recipes
│   └── scenarios/           # accident/closure scenario definitions
├── scripts/                # generate_sandbox_network.py, build_district_network.py,
│                            # generate_demand.py, run_twin_smoke_test.py
├── src/atlas/
│   ├── common/             # config loading, logging, SUMO discovery
│   ├── twin/                # SumoTwinEnv, demand/fleet/scenario modules
│   └── control/             # signal safety shield; classical controllers (Week 3)
├── tests/unit/
├── data/networks/
│   ├── sandbox/             # generated (gitignored), regenerable from scripts/
│   └── manhattan_midtown/   # raw.osm.xml + AUDIT.md committed; everything else regenerated
└── pyproject.toml
```

## Development

```bash
ruff check .        # lint
ruff format .        # format
mypy                  # type-check (strict)
pytest --cov=atlas   # tests + coverage
```

## License

MIT.
