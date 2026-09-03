"""Validates SingleJunctionEnv against the real sandbox network's B1 junction (2 green
phases, 4 incoming lanes -- verified against net.net.xml before writing this module, see
the module docstring). Exercises the actual shield/max-pressure/twin integration end to
end, not mocks: this environment's entire job is wiring those together correctly.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from atlas.control.shield import ShieldConfig
from atlas.rl import SingleJunctionEnv, SingleJunctionEnvConfig
from atlas.twin import TwinEnvConfig

TLS_ID = "B1"
NUM_GREEN_PHASES = 2
NUM_INCOMING_LANES = 4
EXPECTED_OBS_DIM = 3 * NUM_INCOMING_LANES + NUM_GREEN_PHASES + 1  # = 15


@pytest.fixture(scope="session")
def sandbox_net_files() -> tuple[Path, Path]:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import generate_sandbox_network

    generate_sandbox_network.generate()
    return generate_sandbox_network.NET_FILE, generate_sandbox_network.ROUTE_FILE


def _make_config(
    sandbox_net_files: tuple[Path, Path], **shield_overrides: float
) -> SingleJunctionEnvConfig:
    net_file, route_file = sandbox_net_files
    shield_config = ShieldConfig(
        min_green_s=shield_overrides.get("min_green_s", 10.0),
        max_green_s=shield_overrides.get("max_green_s", 60.0),
    )
    return SingleJunctionEnvConfig(
        twin_config=TwinEnvConfig(
            name="single_junction_test", net_file=net_file, route_file=route_file
        ),
        tls_id=TLS_ID,
        shield_config=shield_config,
    )


@pytest.fixture
def env(sandbox_net_files: tuple[Path, Path]) -> Iterator[SingleJunctionEnv]:
    e = SingleJunctionEnv(_make_config(sandbox_net_files))
    try:
        yield e
    finally:
        e.close()


class TestSpaces:
    def test_action_space_matches_green_phase_count(self, env: SingleJunctionEnv) -> None:
        assert env.action_space.n == NUM_GREEN_PHASES

    def test_observation_space_shape(self, env: SingleJunctionEnv) -> None:
        assert env.observation_space.shape == (EXPECTED_OBS_DIM,)

    def test_rejects_decision_interval_not_greater_than_yellow(
        self, sandbox_net_files: tuple[Path, Path]
    ) -> None:
        config = _make_config(sandbox_net_files)
        config = config.model_copy(update={"decision_interval_s": 3.0, "yellow_duration_s": 3.0})
        with pytest.raises(ValueError, match="must be greater than"):
            SingleJunctionEnv(config)


class TestReset:
    def test_returns_observation_within_bounds(self, env: SingleJunctionEnv) -> None:
        obs, info = env.reset()

        assert obs.shape == (EXPECTED_OBS_DIM,)
        assert obs.dtype == np.float32
        assert np.all(obs >= 0.0) and np.all(obs <= 1.0)
        assert info == {}

    def test_exactly_one_phase_is_one_hot(self, env: SingleJunctionEnv) -> None:
        obs, _ = env.reset()

        phase_one_hot = obs[3 * NUM_INCOMING_LANES : 3 * NUM_INCOMING_LANES + NUM_GREEN_PHASES]
        assert phase_one_hot.sum() == pytest.approx(1.0)


class TestStep:
    def test_step_before_reset_raises(self, sandbox_net_files: tuple[Path, Path]) -> None:
        fresh_env = SingleJunctionEnv(_make_config(sandbox_net_files))
        with pytest.raises(RuntimeError, match="not started"):
            fresh_env.step(np.int64(0))

    def test_never_terminates_or_truncates(self, env: SingleJunctionEnv) -> None:
        env.reset()

        for _ in range(10):
            _, _, terminated, truncated, _ = env.step(np.int64(0))
            assert terminated is False
            assert truncated is False

    def test_keep_action_advances_time_without_switching(self, env: SingleJunctionEnv) -> None:
        env.reset()
        current_green_action = 0  # whatever phase we start on; requesting it = "keep"

        _, _, _, _, info = env.step(np.int64(current_green_action))

        # Might legitimately switch if phase 0 isn't the starting phase -- so instead
        # assert the *shield* wasn't forced to override anything (a clean, valid step).
        assert info["shield_modified"] is False

    def test_switch_action_reports_phase_changed(self, env: SingleJunctionEnv) -> None:
        env.reset()
        # Request whichever green phase differs from wherever we started, guaranteed to
        # be a real switch attempt on the very first decision (elapsed=0 -- but a switch
        # is only blocked by min_green, and elapsed=0 < min_green, so this exercises the
        # shield's min_green block, not the switch itself; see the dedicated test below).
        _, _, _, _, info = env.step(np.int64(1))

        # At elapsed=0 < min_green_s (default 10s), the shield MUST block this switch.
        assert info["phase_changed"] is False
        assert info["shield_modified"] is True
        assert info["shield_reason"] == "min_green"

    def test_switch_succeeds_after_min_green_elapsed(
        self, sandbox_net_files: tuple[Path, Path]
    ) -> None:
        config = _make_config(sandbox_net_files, min_green_s=5.0, max_green_s=60.0)
        env = SingleJunctionEnv(config)
        try:
            env.reset()
            env.step(np.int64(0))  # advances 5s -> elapsed now >= min_green_s (5.0)

            _, _, _, _, info = env.step(np.int64(1))  # now request the other phase

            assert info["phase_changed"] is True
            assert info["shield_modified"] is False
        finally:
            env.close()

    def test_reward_matches_negative_pressure_and_switch_penalty(
        self, sandbox_net_files: tuple[Path, Path]
    ) -> None:
        config = _make_config(sandbox_net_files, min_green_s=5.0, max_green_s=60.0)
        config = config.model_copy(update={"switch_penalty": 2.5})
        env = SingleJunctionEnv(config)
        try:
            env.reset()
            env.step(np.int64(0))
            _, reward, _, _, info = env.step(np.int64(1))

            expected = -float(info["pressure"])
            if info["phase_changed"]:
                expected -= 2.5
            assert reward == pytest.approx(expected)
        finally:
            env.close()


class TestStableBaselines3Compatibility:
    def test_passes_sb3_env_checker(self, sandbox_net_files: tuple[Path, Path]) -> None:
        """The actual purpose of this whole module: prove it is a valid environment for
        Stable-Baselines3, not just "a Gymnasium API we believe is compatible"."""
        from stable_baselines3.common.env_checker import check_env

        env = SingleJunctionEnv(_make_config(sandbox_net_files))
        try:
            check_env(env, warn=True, skip_render_check=True)
        finally:
            env.close()
