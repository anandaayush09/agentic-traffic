"""Validates MultiAgentSignalEnv against all 4 real sandbox junctions (A1, B0, B1, B2 --
uniformly 2 green phases, 3-4 incoming lanes each, verified against net.net.xml before
writing this module). Exercises padding/validity flags and simultaneous multi-junction
phase switching, the two behaviors unique to the multi-agent generalization of Week 3's
SingleJunctionEnv.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from atlas.control.shield import ShieldConfig
from atlas.rl import MultiAgentSignalEnv, MultiAgentSignalEnvConfig
from atlas.twin import TwinEnvConfig

TLS_IDS = ["A1", "B0", "B1", "B2"]
MAX_LANES = 4  # B1's actual lane count; A1/B0/B2 have 3, so 1 padding slot each
MAX_GREEN_PHASES = 2  # every junction's actual count
OBS_DIM = 4 * MAX_LANES + MAX_GREEN_PHASES + 2  # = 20 (+1 elapsed, +1 Week 7 corridor hint)


@pytest.fixture(scope="session")
def sandbox_net_route() -> tuple[Path, Path]:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import generate_sandbox_network

    generate_sandbox_network.generate()
    return generate_sandbox_network.NET_FILE, generate_sandbox_network.ROUTE_FILE


def _make_config(
    net_route: tuple[Path, Path],
    max_lanes: int = MAX_LANES,
    max_green_phases: int = MAX_GREEN_PHASES,
) -> MultiAgentSignalEnvConfig:
    net_file, route_file = net_route
    return MultiAgentSignalEnvConfig(
        twin_config=TwinEnvConfig(
            name="multi_agent_test", net_file=net_file, route_file=route_file
        ),
        tls_ids=TLS_IDS,
        max_lanes=max_lanes,
        max_green_phases=max_green_phases,
        shield_config=ShieldConfig(min_green_s=10.0, max_green_s=60.0),
    )


@pytest.fixture
def env(sandbox_net_route: tuple[Path, Path]) -> Iterator[MultiAgentSignalEnv]:
    e = MultiAgentSignalEnv(_make_config(sandbox_net_route))
    try:
        yield e
    finally:
        e.close()


class TestConstruction:
    def test_agents_match_configured_tls_ids(self, env: MultiAgentSignalEnv) -> None:
        assert set(env.agents) == set(TLS_IDS)

    def test_observation_and_action_dims(self, env: MultiAgentSignalEnv) -> None:
        assert env.observation_dim == OBS_DIM
        assert env.action_dim == MAX_GREEN_PHASES

    def test_rejects_max_lanes_too_small_for_a_real_junction(
        self, sandbox_net_route: tuple[Path, Path]
    ) -> None:
        # B1 has 4 incoming lanes; max_lanes=3 must be rejected, not silently truncated.
        with pytest.raises(ValueError, match="exceeding max_lanes"):
            MultiAgentSignalEnv(_make_config(sandbox_net_route, max_lanes=3))


class TestResetContract:
    def test_returns_observation_for_every_agent(self, env: MultiAgentSignalEnv) -> None:
        obs = env.reset()

        assert set(obs.keys()) == set(TLS_IDS)
        for agent_obs in obs.values():
            assert agent_obs.shape == (OBS_DIM,)
            assert agent_obs.dtype == np.float32
            assert np.all(agent_obs >= 0.0) and np.all(agent_obs <= 1.0)

    def test_padding_lane_marked_invalid_for_three_lane_junctions(
        self, env: MultiAgentSignalEnv
    ) -> None:
        obs = env.reset()

        # A1 has 3 real lanes (< MAX_LANES=4): lane slots are [q,occ,speed,valid] x 4,
        # so the 4th (padding) slot's valid flag is features[15] (index 3*4+3).
        a1_obs = obs["A1"]
        real_lane_valid_flags = [a1_obs[i * 4 + 3] for i in range(3)]
        padding_valid_flag = a1_obs[3 * 4 + 3]

        assert all(flag == 1.0 for flag in real_lane_valid_flags)
        assert padding_valid_flag == 0.0

    def test_full_lane_junction_has_no_padding(self, env: MultiAgentSignalEnv) -> None:
        obs = env.reset()

        # B1 has exactly MAX_LANES=4 real lanes -- every slot should be valid.
        b1_obs = obs["B1"]
        valid_flags = [b1_obs[i * 4 + 3] for i in range(MAX_LANES)]

        assert all(flag == 1.0 for flag in valid_flags)


class TestStepContract:
    def test_step_before_reset_raises(self, sandbox_net_route: tuple[Path, Path]) -> None:
        fresh_env = MultiAgentSignalEnv(_make_config(sandbox_net_route))
        with pytest.raises(RuntimeError, match="not started"):
            fresh_env.step({tls_id: 0 for tls_id in TLS_IDS})

    def test_rejects_missing_agent_actions(self, env: MultiAgentSignalEnv) -> None:
        env.reset()

        with pytest.raises(ValueError, match="Missing actions"):
            env.step({"A1": 0})  # missing B0, B1, B2

    def test_returns_dicts_for_every_agent(self, env: MultiAgentSignalEnv) -> None:
        env.reset()

        obs, rewards, terminations, truncations, infos = env.step({t: 0 for t in TLS_IDS})

        for d in (obs, rewards, terminations, truncations, infos):
            assert set(d.keys()) == set(TLS_IDS)
        for t in TLS_IDS:
            assert terminations[t] is False
            assert truncations[t] is False

    def test_multiple_junctions_can_switch_simultaneously(
        self, sandbox_net_route: tuple[Path, Path]
    ) -> None:
        config = _make_config(sandbox_net_route)
        config = config.model_copy(
            update={"shield_config": ShieldConfig(min_green_s=5.0, max_green_s=60.0)}
        )
        env = MultiAgentSignalEnv(config)
        try:
            env.reset()
            env.step({t: 0 for t in TLS_IDS})  # advance 5s past min_green

            _, _, _, _, infos = env.step(
                {t: 1 for t in TLS_IDS}
            )  # request the other phase, all at once

            assert all(infos[t]["phase_changed"] for t in TLS_IDS)
            assert all(not infos[t]["shield_modified"] for t in TLS_IDS)
        finally:
            env.close()

    def test_action_beyond_green_phase_count_wraps_via_modulo(
        self, sandbox_net_route: tuple[Path, Path]
    ) -> None:
        # max_green_phases=3 > every junction's real count of 2, so action=2 must behave
        # identically to action=0 (2 % 2 == 0) rather than crashing.
        config = _make_config(sandbox_net_route, max_green_phases=3)
        env = MultiAgentSignalEnv(config)
        try:
            env.reset()

            _, _, _, _, infos = env.step({t: 2 for t in TLS_IDS})

            for t in TLS_IDS:
                assert infos[t]["shield_reason"] != "invalid_phase"
        finally:
            env.close()


class TestCorridorHints:
    """Week 7: an advisory feature appended to the observation vector -- see
    docs/design/04_agent_architecture.md section 4.2."""

    def test_defaults_to_zero_when_never_set(self, env: MultiAgentSignalEnv) -> None:
        observations = env.reset()
        for tls_id in TLS_IDS:
            assert observations[tls_id][-1] == pytest.approx(0.0)

    def test_set_hints_appear_in_the_next_observation(self, env: MultiAgentSignalEnv) -> None:
        env.reset()
        env.set_corridor_hints({"B1": 0.75})

        observations, _, _, _, _ = env.step({t: 0 for t in TLS_IDS})

        assert observations["B1"][-1] == pytest.approx(0.75)
        assert observations["A1"][-1] == pytest.approx(0.0)  # untouched agent unaffected

    def test_partial_update_does_not_clear_previously_set_hints(
        self, env: MultiAgentSignalEnv
    ) -> None:
        env.reset()
        env.set_corridor_hints({"B0": 0.5, "B1": 0.2})
        env.set_corridor_hints({"B1": 0.9})  # only updates B1

        observations, _, _, _, _ = env.step({t: 0 for t in TLS_IDS})

        assert observations["B0"][-1] == pytest.approx(0.5)
        assert observations["B1"][-1] == pytest.approx(0.9)
