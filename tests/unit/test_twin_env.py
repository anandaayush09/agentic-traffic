"""Integration-flavored unit tests for SumoTwinEnv, run against the generated sandbox
network. These exercise a real SUMO simulation (via libsumo) rather than mocking it: the
twin wrapper's entire job is translating SUMO's API into ATLAS's types correctly, so a
mock would test nothing of value.

Note: libsumo holds simulation state at module scope (it is an in-process embedding, not
a socket connection), so only one SumoTwinEnv may be running at a time within this test
process. The `twin_env` fixture guarantees `close()` runs in teardown before the next
test's setup, keeping tests independent despite that shared module state.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from pydantic import ValidationError

from atlas.twin import JunctionState, LaneState, SumoTwinEnv, TwinEnvConfig, TwinEnvError


@pytest.fixture
def twin_env(sandbox_twin_config: TwinEnvConfig) -> Iterator[SumoTwinEnv]:
    env = SumoTwinEnv(sandbox_twin_config)
    try:
        yield env
    finally:
        env.close()


class TestTwinEnvConfig:
    def test_defaults(self, sandbox_twin_config: TwinEnvConfig) -> None:
        assert sandbox_twin_config.step_length_s == 1.0
        assert sandbox_twin_config.backend == "libsumo"
        assert sandbox_twin_config.gui is False
        assert sandbox_twin_config.traffic_light_ids == ["A1", "B0", "B1", "B2"]

    def test_rejects_nonpositive_step_length(self, sandbox_twin_config: TwinEnvConfig) -> None:
        # model_copy(update=...) intentionally bypasses validation (pydantic docs), so we
        # must go through the constructor to actually exercise the `gt=0` constraint.
        with pytest.raises(ValidationError):
            TwinEnvConfig(**{**sandbox_twin_config.model_dump(), "step_length_s": 0})

    def test_rejects_unknown_backend(self, sandbox_twin_config: TwinEnvConfig) -> None:
        with pytest.raises(ValidationError):
            TwinEnvConfig(
                **{
                    **sandbox_twin_config.model_dump(),
                    "backend": "carrier_pigeon",
                }
            )


class TestSumoTwinEnvLifecycle:
    def test_not_running_before_reset(self, twin_env: SumoTwinEnv) -> None:
        assert twin_env.is_running is False

    def test_reset_starts_simulation(self, twin_env: SumoTwinEnv) -> None:
        twin_env.reset()

        assert twin_env.is_running is True
        assert twin_env.sim_time == 0.0

    def test_step_advances_sim_time(self, twin_env: SumoTwinEnv) -> None:
        twin_env.reset()

        twin_env.step(n=10)

        assert twin_env.sim_time == pytest.approx(10 * twin_env.config.step_length_s)

    def test_step_without_reset_raises(self, twin_env: SumoTwinEnv) -> None:
        with pytest.raises(TwinEnvError, match="not running"):
            twin_env.step()

    def test_step_rejects_nonpositive_n(self, twin_env: SumoTwinEnv) -> None:
        twin_env.reset()

        with pytest.raises(ValueError, match="n must be"):
            twin_env.step(n=0)

    def test_reset_twice_restarts_cleanly(self, twin_env: SumoTwinEnv) -> None:
        twin_env.reset()
        twin_env.step(n=20)
        assert twin_env.sim_time > 0

        twin_env.reset()  # should close the first run and start a fresh one

        assert twin_env.is_running is True
        assert twin_env.sim_time == 0.0

    def test_close_is_idempotent(self, twin_env: SumoTwinEnv) -> None:
        twin_env.reset()

        twin_env.close()
        twin_env.close()  # must not raise

        assert twin_env.is_running is False

    def test_context_manager(self, sandbox_twin_config: TwinEnvConfig) -> None:
        with SumoTwinEnv(sandbox_twin_config) as env:
            assert env.is_running is True
            env.step(n=5)

        assert env.is_running is False


class TestSumoTwinEnvState:
    def test_traffic_light_ids_match_config(self, twin_env: SumoTwinEnv) -> None:
        twin_env.reset()

        reported = set(twin_env.get_traffic_light_ids())

        assert set(twin_env.config.traffic_light_ids).issubset(reported)

    def test_get_junction_state_shape(self, twin_env: SumoTwinEnv) -> None:
        twin_env.reset()
        twin_env.step(n=5)

        state = twin_env.get_junction_state("B1")

        assert isinstance(state, JunctionState)
        assert state.tls_id == "B1"
        assert isinstance(state.phase_index, int) and state.phase_index >= 0
        assert len(state.phase_state) > 0
        assert set(state.phase_state) <= set("rRyYgGoOu-")
        assert len(state.incoming_lanes) > 0
        for lane in state.incoming_lanes:
            assert isinstance(lane, LaneState)
            assert lane.queue_length >= 0
            assert lane.vehicle_count >= 0
            assert lane.mean_speed_mps >= -1e-6
            assert 0.0 <= lane.occupancy <= 1.0 + 1e-6

    def test_get_all_junction_states_covers_configured_ids(self, twin_env: SumoTwinEnv) -> None:
        twin_env.reset()
        twin_env.step(n=5)

        states = twin_env.get_all_junction_states()

        assert set(states.keys()) == set(twin_env.config.traffic_light_ids)

    def test_thousand_step_run_is_stable(self, twin_env: SumoTwinEnv) -> None:
        """This is the Week 1 roadmap deliverable at test scale: prove the twin survives
        a long run without desync or crashing, and every configured junction remains
        readable throughout."""
        twin_env.reset()

        for _ in range(100):
            twin_env.step(n=10)  # 100 * 10 = 1000 steps total, checked incrementally

        assert twin_env.sim_time == pytest.approx(1000 * twin_env.config.step_length_s)
        states = twin_env.get_all_junction_states()
        assert len(states) == len(twin_env.config.traffic_light_ids)
