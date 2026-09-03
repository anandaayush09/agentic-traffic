# ATLAS — Agentic Traffic Learning & Adaptive Signaling

> **Status: Week 2 of 12 (in progress) — Network Audit & Real District Acquisition.**
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

**Week 2 (in progress) — real district network + demand:**
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
- Not yet done: fleet mix (buses/trucks/emergency vehicles) and the scenario engine
  (accident/closure/rush-hour disruption classes) — see
  [`docs/design/13_roadmap.md`](docs/design/13_roadmap.md), Week 2.

Everything else described in `docs/design/` (the control plane, event bus, cognitive
plane, perception, forecasting, dashboard, deployment) is designed but **not yet built**
— see [`docs/design/13_roadmap.md`](docs/design/13_roadmap.md) for the week-by-week plan.

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

# Run the test suite
pytest
```

## Repository layout

See [`docs/design/09_repository_design.md`](docs/design/09_repository_design.md) for the
full target layout and the rationale behind it. Current state:

```
agentic-traffic/
├── docs/design/          # frozen architecture (15 files, Parts 1-14 + exec summary)
├── configs/network/       # declarative YAML network/twin configs
├── scripts/                # generate_sandbox_network.py, build_district_network.py,
│                            # run_twin_smoke_test.py
├── src/atlas/
│   ├── common/             # config loading, logging, SUMO discovery
│   └── twin/                # SumoTwinEnv: the digital-twin wrapper
├── tests/unit/
├── data/networks/
│   ├── sandbox/             # generated (gitignored), regenerable from scripts/
│   └── manhattan_midtown/   # raw.osm.xml + AUDIT.md committed; net.net.xml regenerated
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
