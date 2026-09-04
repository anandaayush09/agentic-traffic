"""Validates Ray rollout workers against the sandbox network's 4 junctions: the Week 4
roadmap's explicit "Ray worker determinism-with-seed test", plus a genuine multi-process
isolation check (two actors running concurrently must not share state -- meaningful given
libsumo's single-simulation-per-process constraint, see rollout.py's module docstring).

Every test here is marked `ray_isolated` and excluded from the default `pytest` run (see
pyproject.toml's marker registration for why): `ray.init()` in this process permanently
breaks subsequent libsumo imports in that SAME process, even after `ray.shutdown()` --
verified directly (not guessed at) by reproducing the failure with test_twin_env.py's
already-passing libsumo tests when run together in one pytest session. Run this file with
`pytest -m ray_isolated`, in its own process, never combined with libsumo-based tests.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import ray

from atlas.control.shield import ShieldConfig
from atlas.rl import MultiAgentSignalEnvConfig
from atlas.rl.rollout import RolloutWorker
from atlas.twin import TwinEnvConfig

pytestmark = pytest.mark.ray_isolated

TLS_IDS = ["A1", "B0", "B1", "B2"]
ROLLOUT_STEPS = 20


@pytest.fixture(scope="session")
def sandbox_net_route() -> tuple[Path, Path]:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import generate_sandbox_network

    generate_sandbox_network.generate()
    return generate_sandbox_network.NET_FILE, generate_sandbox_network.ROUTE_FILE


@pytest.fixture(scope="session", autouse=True)
def ray_session() -> Iterator[None]:
    # No num_cpus override: constraining it without also setting object_store_memory let
    # Ray auto-size the object store below its own hard minimum (a real bug hit during
    # development -- ValueError: object store memory 61481779 < minimum 78643200). Tests
    # here only ever spin up 1-2 actors regardless of the cluster's advertised CPU count,
    # so the default (auto-detected) sizing is fine.
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


def _make_config(net_route: tuple[Path, Path]) -> MultiAgentSignalEnvConfig:
    net_file, route_file = net_route
    return MultiAgentSignalEnvConfig(
        # backend="traci": libsumo fails to import inside a Ray worker process on
        # Windows (a real native DLL collision, verified -- see rollout.py's module
        # docstring). Not the frozen architecture's default anywhere else.
        twin_config=TwinEnvConfig(
            name="rollout_test", net_file=net_file, route_file=route_file, backend="traci"
        ),
        tls_ids=TLS_IDS,
        max_lanes=4,
        max_green_phases=2,
        shield_config=ShieldConfig(min_green_s=10.0, max_green_s=60.0),
    )


class TestRolloutDeterminism:
    def test_same_seed_produces_identical_fingerprint(
        self, sandbox_net_route: tuple[Path, Path]
    ) -> None:
        config = _make_config(sandbox_net_route)
        worker_a = RolloutWorker.remote(config)
        worker_b = RolloutWorker.remote(config)
        try:
            result_a, result_b = ray.get(
                [
                    worker_a.collect_random_rollout.remote(ROLLOUT_STEPS, seed=42),
                    worker_b.collect_random_rollout.remote(ROLLOUT_STEPS, seed=42),
                ]
            )
            assert result_a["fingerprint"] == result_b["fingerprint"]
            assert result_a["total_reward"] == pytest.approx(result_b["total_reward"])
        finally:
            ray.get([worker_a.close.remote(), worker_b.close.remote()])

    def test_different_seeds_produce_different_fingerprints(
        self, sandbox_net_route: tuple[Path, Path]
    ) -> None:
        config = _make_config(sandbox_net_route)
        worker_a = RolloutWorker.remote(config)
        worker_b = RolloutWorker.remote(config)
        try:
            result_a, result_b = ray.get(
                [
                    worker_a.collect_random_rollout.remote(ROLLOUT_STEPS, seed=1),
                    worker_b.collect_random_rollout.remote(ROLLOUT_STEPS, seed=2),
                ]
            )
            assert result_a["fingerprint"] != result_b["fingerprint"]
        finally:
            ray.get([worker_a.close.remote(), worker_b.close.remote()])


class TestRolloutParallelExecution:
    def test_two_workers_run_concurrently_without_cross_contamination(
        self, sandbox_net_route: tuple[Path, Path]
    ) -> None:
        """Each actor is its own process with its own libsumo instance -- running two
        concurrently, each must report exactly its own agent set and step count, not a
        mix of both."""
        config = _make_config(sandbox_net_route)
        workers = [RolloutWorker.remote(config) for _ in range(2)]
        try:
            results = ray.get(
                [
                    w.collect_random_rollout.remote(ROLLOUT_STEPS, seed=i)
                    for i, w in enumerate(workers)
                ]
            )
            for r in results:
                assert r["n_steps"] == ROLLOUT_STEPS
                assert r["num_agents"] == len(TLS_IDS)
                assert len(r["fingerprint"]) == ROLLOUT_STEPS
        finally:
            ray.get([w.close.remote() for w in workers])

    def test_reset_and_step_passthrough(self, sandbox_net_route: tuple[Path, Path]) -> None:
        config = _make_config(sandbox_net_route)
        worker = RolloutWorker.remote(config)
        try:
            obs = ray.get(worker.reset.remote(seed=42))
            assert set(obs.keys()) == set(TLS_IDS)

            step_result = ray.get(worker.step.remote({t: 0 for t in TLS_IDS}))
            _, rewards, terminations, truncations, infos = step_result
            assert set(rewards.keys()) == set(TLS_IDS)
        finally:
            ray.get(worker.close.remote())
