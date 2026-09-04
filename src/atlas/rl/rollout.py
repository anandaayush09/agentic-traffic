"""Ray Core rollout workers: parallelize MultiAgentSignalEnv stepping across worker
processes, the Week 4 roadmap deliverable ahead of distributed MAPPO training (Week 5;
docs/design/13_roadmap.md, Week 4; docs/design/07_tech_stack.md section 7.2 -- "Ray Core
used only for parallel rollout workers", not RLlib).

One `RolloutWorker` actor = one process = one SUMO simulation. This isn't an arbitrary
design choice: libsumo is an in-process embedding with module-level state (verified back
in Week 1 -- only one live simulation per Python process), so it cannot be shared or
pooled across concurrent rollouts. Ray's actor model gives each actor its own OS process
by default, which is exactly the isolation libsumo requires; this is why the parallel
rollout collection this project needs is naturally expressed as Ray actors rather than,
say, a thread pool (which would share one process, and therefore one libsumo instance).

**Backend caveat, verified before writing this note (not guessed at): `libsumo` (the
frozen architecture's default backend, Part 7.2) fails to import inside a Ray worker
process on this Windows machine** -- `ImportError: DLL load failed while importing
_libsumo: The specified procedure could not be found` (Windows exception code
0xc0000139, STATUS_ENTRYPOINT_NOT_FOUND). Isolated by direct testing: `PATH` is byte-
for-byte identical between the Ray driver and worker processes (rules out search-order
shadowing); `traci` (pure Python, no compiled extension of its own) imports and runs a
full rollout successfully in the same worker where `libsumo` fails. This points to a
native DLL/symbol collision between libsumo's compiled extension and something Ray's own
Cython/C++ components load into the process first -- not something fixable from this
codebase. Every `TwinEnvConfig` constructed for a `RolloutWorker` in this project
therefore uses `backend="traci"`, NOT libsumo -- a deliberate, documented, platform-
specific exception to the frozen architecture's default, not a silent regression.
Consequence: throughput measured through Ray workers on this machine is a *pessimistic*
lower bound (traci's socket IPC overhead vs. libsumo's in-process calls) -- it is not
evidence about what libsumo-in-Ray-workers would achieve on Linux/Kaggle, where this
exact collision may not reproduce at all (worth re-testing there before trusting this
number as libsumo's ceiling). Everywhere else in this codebase (SingleJunctionEnv,
MultiAgentSignalEnv used directly, all existing tests, training scripts) still defaults
to libsumo, per the frozen architecture.

No trained policy exists yet -- that is Week 5's from-scratch MAPPO. `collect_random_rollout`
steps with random actions, useful for exactly two things right now: a throughput benchmark
(scripts/benchmark_ray_rollout.py) and a determinism-with-seed test (proving the same seed
reproduces the same trajectory, a precondition for reproducible training later).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import ray

from atlas.rl.multi_agent_env import MultiAgentSignalEnv, MultiAgentSignalEnvConfig


@ray.remote
class RolloutWorker:
    """One Ray actor wrapping one MultiAgentSignalEnv instance."""

    def __init__(self, config: MultiAgentSignalEnvConfig) -> None:
        self._env = MultiAgentSignalEnv(config)

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        return self._env.reset(seed=seed)

    def step(
        self, actions: dict[str, int]
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        return self._env.step(actions)

    def collect_random_rollout(self, n_steps: int, seed: int | None = None) -> dict[str, Any]:
        """Steps the environment `n_steps` times with actions sampled from a
        seed-controlled RNG (independent of the twin's own SUMO seed, so the same
        `seed` here always produces the same action sequence regardless of any change
        to demand/traffic randomness). Returns:
        - `fingerprint`: the per-step sum of every agent's pressure, a compact
          deterministic signature of the whole trajectory -- two rollouts with the same
          seed must produce identical fingerprints; different seeds should not (in
          practice, virtually never do).
        - `wall_clock_s` / `n_steps`: raw timing, for the throughput benchmark.
        """
        rng = np.random.default_rng(seed)
        self._env.reset(seed=seed)

        start = time.perf_counter()
        total_reward = 0.0
        fingerprint: list[float] = []
        for _ in range(n_steps):
            actions = {
                agent_id: int(rng.integers(0, self._env.action_dim))
                for agent_id in self._env.agents
            }
            _, rewards, _, _, infos = self._env.step(actions)
            total_reward += sum(rewards.values())
            fingerprint.append(sum(infos[a]["pressure"] for a in self._env.agents))
        wall_clock_s = time.perf_counter() - start

        return {
            "n_steps": n_steps,
            "total_reward": total_reward,
            "fingerprint": fingerprint,
            "wall_clock_s": wall_clock_s,
            "num_agents": len(self._env.agents),
        }

    def close(self) -> None:
        self._env.close()
