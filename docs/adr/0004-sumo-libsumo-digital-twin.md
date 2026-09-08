# ADR-0004: SUMO + libsumo as the digital twin

**Status:** Accepted (design Part 3; built Week 1, commit `ae7f5e4`)

## Context

Every other component (control, evaluation, the cognitive plane's what-if tool) depends on a
microscopic traffic simulator that is fast enough for RL sample throughput on free-tier
compute (Colab/Kaggle), realistic enough to support OSM import, emissions, and public-transit
modeling, and has enough ecosystem/tooling to be a credible sim-to-real story rather than a
toy.

## Decision

SUMO as the simulator, accessed via **libsumo** (SUMO's in-process C++ binding) rather than
**TraCI** (its socket-based RPC protocol) wherever possible — libsumo removes the
socket-serialization bottleneck that otherwise dominates step time at RL training throughput.

## Alternatives considered

- **CityFlow:** faster raw stepping, but no emissions model, no public-transit support, and a
  largely dead ecosystem — rejected because both PT priority (a stated project goal) and
  ecosystem credibility matter more than raw speed here.
- **Vissim / CARLA:** rejected — commercial licensing (Vissim) or wrong simulation domain
  (CARLA is built for autonomous-vehicle perception/control, not city-scale traffic signal
  optimization).

## Consequences

- **libsumo cannot be imported inside a Ray worker process on Windows** — a real native-DLL
  collision found in Week 4 (verified: `traci` works fine in the same worker where `libsumo`
  fails). Documented, platform-specific exception: any `TwinEnvConfig` used inside a
  `RolloutWorker` on this dev machine must set `backend="traci"`; the resulting throughput
  numbers are a pessimistic lower bound, not evidence of libsumo's real ceiling on
  Kaggle/Linux.
- Two more import-ordering collisions traced to the same root cause (native extensions
  fighting over process-global state) were found later: `import ray` before `import libsumo`
  permanently breaks libsumo in that process (Week 4), and `import mlflow` before `import
  libsumo` breaks it too, via a transitive pyarrow conflict (Week 5). Both fixed by import
  ordering, not architecture changes, and documented in each affected module.
- `SumoTwinEnv.connection` is exposed as a deliberate escape hatch to the raw libsumo/TraCI
  handle for lane/edge-level control the wrapper doesn't itself expose (used by the Week 2
  scenario engine and Week 11's evaluation runner) — a conscious choice to keep the wrapper
  thin rather than growing a method for every SUMO API surface a future feature might need.
