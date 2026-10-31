"""Validates the RL-policy plumbing that is built ahead of a trained checkpoint: loading both
checkpoint formats, refusing a policy that doesn't fit, running a policy through the live twin
and the evaluation runner (always via the safety shield), and the wall-clock training budget.

Every policy here is a RANDOM-INIT network -- no training has happened. These tests prove the
plumbing, not that RL controls traffic well; that needs a real checkpoint and the evaluation.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

from atlas.common.config import project_root
from atlas.control.shield import ShieldConfig
from atlas.evaluation.runner import run_scenario_cell
from atlas.marl import (
    ActorCritic,
    TrainingRunConfig,
    check_policy_fits,
    load_policy,
    run_training,
)
from atlas.marl.policy_io import LoadedPolicy
from atlas.marl.training import checkpoint_path, load_checkpoint
from atlas.rl import MultiAgentSignalEnv, MultiAgentSignalEnvConfig
from atlas.twin import TwinEnvConfig
from atlas.twin.controlled_twin import CORRIDOR_TLS_IDS, DISTRICT_TLS_IDS, ControlledTwin
from atlas.twin.incident_effects import RoadIndex

NETWORK_DIR = project_root() / "data" / "networks" / "manhattan_midtown"
NET_FILE = NETWORK_DIR / "net.net.xml"
ROUTE_FILE = NETWORK_DIR / "demand" / "pm_peak.rou.xml"
VTYPES_FILE = NETWORK_DIR / "vtypes.add.xml"

requires_network = pytest.mark.skipif(
    not (NET_FILE.is_file() and ROUTE_FILE.is_file()), reason="manhattan_midtown network not built"
)

# MultiAgentSignalEnv with max_lanes=10, max_green_phases=2 -> 4*10 + 2 + 2 = 44 inputs.
OBS_DIM, ACTION_DIM = 44, 2


def _random_net(num_agents: int = 22, obs_dim: int = OBS_DIM) -> ActorCritic:
    torch.manual_seed(0)
    return ActorCritic(obs_dim, ACTION_DIM, num_agents)


def _policy(net: ActorCritic | None = None, source: str = "random.pt") -> LoadedPolicy:
    net = net or _random_net()
    net.eval()
    return LoadedPolicy(
        net=net,
        source=Path(source),
        iteration=None,
        obs_dim=net.obs_dim,
        action_dim=net.action_dim,
        trained_agents=net.num_agents,
    )


class TestLoadPolicy:
    def test_loads_a_plain_state_dict(self, tmp_path: Path) -> None:
        net = _random_net()
        path = tmp_path / "mappo_district.pt"
        torch.save(net.state_dict(), path)

        loaded = load_policy(path)

        assert (loaded.obs_dim, loaded.action_dim, loaded.trained_agents) == (OBS_DIM, 2, 22)
        assert loaded.iteration is None
        obs = torch.randn(7, OBS_DIM)
        assert torch.equal(loaded.net.actor(obs), net.actor(obs))  # same weights

    def test_loads_a_resumable_checkpoint_and_reports_its_iteration(self, tmp_path: Path) -> None:
        net = _random_net(num_agents=5)
        path = tmp_path / "run_latest.pt"
        torch.save({"iteration": 41, "net_state_dict": net.state_dict(), "optimizer": {}}, path)

        loaded = load_policy(path)

        assert loaded.iteration == 41
        assert loaded.trained_agents == 5
        assert "iteration 41" in loaded.description

    def test_loaded_network_is_in_eval_mode(self, tmp_path: Path) -> None:
        path = tmp_path / "p.pt"
        torch.save(_random_net().state_dict(), path)

        assert load_policy(path).net.training is False

    def test_a_missing_file_is_a_clear_error(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="No policy file"):
            load_policy(tmp_path / "nope.pt")

    def test_a_file_that_is_not_a_policy_is_rejected(self, tmp_path: Path) -> None:
        garbage = tmp_path / "garbage.pt"
        garbage.write_bytes(b"not a torch file")
        wrong = tmp_path / "wrong.pt"
        torch.save({"something": torch.zeros(3)}, wrong)

        with pytest.raises(ValueError, match="not a loadable"):
            load_policy(garbage)
        with pytest.raises(ValueError, match="not an ActorCritic"):
            load_policy(wrong)


class TestCheckPolicyFits:
    def test_matching_widths_pass(self) -> None:
        check_policy_fits(_policy(), OBS_DIM, ACTION_DIM)

    def test_a_different_observation_or_action_width_is_a_clear_error(self) -> None:
        small = _policy(_random_net(obs_dim=12))

        with pytest.raises(ValueError, match="max_lanes"):
            check_policy_fits(small, OBS_DIM, ACTION_DIM)
        with pytest.raises(ValueError, match="actions"):
            check_policy_fits(_policy(), OBS_DIM, 3)


@pytest.fixture(scope="module")
def index() -> RoadIndex:
    return RoadIndex.from_net_file(NET_FILE)


@requires_network
class TestRlModeInTheLiveTwin:
    @pytest.fixture
    def twin(self, index: RoadIndex) -> Iterator[ControlledTwin]:
        t = ControlledTwin(NET_FILE, ROUTE_FILE, [VTYPES_FILE], road_index=index, policy=_policy())
        t.reset(seed=42)
        t.step_decisions(12)
        try:
            yield t
        finally:
            t.close()

    def test_rl_is_offered_only_when_a_policy_was_supplied(
        self, twin: ControlledTwin, index: RoadIndex
    ) -> None:
        plain = ControlledTwin(NET_FILE, ROUTE_FILE, [VTYPES_FILE], road_index=index)

        assert twin.available_modes == ("fixed_time", "actuated", "max_pressure", "rl")
        assert plain.available_modes == ("fixed_time", "actuated", "max_pressure")
        with pytest.raises(ValueError, match="mode must be one of"):
            plain.set_mode(CORRIDOR_TLS_IDS[0], "rl")

    def test_a_junction_runs_the_policy_through_the_shield(self, twin: ControlledTwin) -> None:
        for tls_id in CORRIDOR_TLS_IDS:
            twin.set_mode(tls_id, "rl")

        twin.step_decisions(10)  # 50 s of the random policy driving 5 junctions

        states = twin.junction_states()
        assert all(states[j]["mode"] == "rl" for j in CORRIDOR_TLS_IDS)
        # The shield enforces legality whatever the policy asks: every phase stays a real one.
        assert all(int(states[j]["phase"]) >= 0 for j in CORRIDOR_TLS_IDS)

    def test_the_policy_can_drive_every_one_of_the_22_junctions(self, twin: ControlledTwin) -> None:
        for tls_id in DISTRICT_TLS_IDS:
            twin.set_mode(tls_id, "rl")

        twin.step_decisions(6)

        assert {s["mode"] for s in twin.junction_states().values()} == {"rl"}

    def test_a_policy_with_the_wrong_widths_is_refused_when_the_twin_starts(
        self, index: RoadIndex
    ) -> None:
        wrong = _policy(_random_net(obs_dim=12))
        twin = ControlledTwin(NET_FILE, ROUTE_FILE, [VTYPES_FILE], road_index=index, policy=wrong)

        with pytest.raises(ValueError, match="observation width"):
            twin.reset(seed=1)


@requires_network
class TestControlScope:
    def test_a_corridor_scoped_twin_leaves_the_other_signals_alone(self, index: RoadIndex) -> None:
        twin = ControlledTwin(
            NET_FILE, ROUTE_FILE, [VTYPES_FILE], road_index=index, tls_ids=CORRIDOR_TLS_IDS
        )
        twin.reset(seed=3)
        try:
            twin.step_decisions(4)

            assert set(twin.junction_states()) == set(CORRIDOR_TLS_IDS)
            with pytest.raises(KeyError, match="not an ATLAS-controlled"):
                twin.set_mode("42428283", "max_pressure")  # a real junction, just not in scope
        finally:
            twin.close()

    def test_the_default_scope_is_all_22_real_signals(self, index: RoadIndex) -> None:
        twin = ControlledTwin(NET_FILE, ROUTE_FILE, [VTYPES_FILE], road_index=index)

        assert len(twin.tls_ids) == 22
        assert set(CORRIDOR_TLS_IDS) <= set(twin.tls_ids)
        twin.reset(seed=3)
        try:
            twin.step_decisions(3)  # constructing 22 shields + deciding for each works
            assert len(twin.junction_states()) == 22
        finally:
            twin.close()


@requires_network
class TestRlInTheEvaluationRunner:
    def _cell(self, tmp_path: Path, **kwargs: object) -> object:
        return run_scenario_cell(
            mode="rl",
            tls_ids=list(CORRIDOR_TLS_IDS),
            net_file=NET_FILE,
            route_file=ROUTE_FILE,
            seed=1,
            num_decisions=12,
            tripinfo_output=tmp_path / "tripinfo.xml",
            additional_files=[VTYPES_FILE],
            **kwargs,  # type: ignore[arg-type]
        )

    def test_rl_without_a_policy_still_fails_loudly(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="requires net and obs"):
            self._cell(tmp_path)

    def test_rl_with_a_policy_runs_a_real_episode_and_reports_metrics(self, tmp_path: Path) -> None:
        metrics = self._cell(tmp_path, policy=_policy())

        assert metrics.mean_waiting_time_s >= 0.0  # type: ignore[attr-defined]


@requires_network
class TestTrainingTimeBudget:
    def _env(self) -> MultiAgentSignalEnv:
        return MultiAgentSignalEnv(
            MultiAgentSignalEnvConfig(
                twin_config=TwinEnvConfig(
                    name="budget_test",
                    net_file=NET_FILE,
                    route_file=ROUTE_FILE,
                    additional_files=[VTYPES_FILE],
                ),
                tls_ids=["42430333", "42430338"],
                max_lanes=10,
                max_green_phases=2,
                shield_config=ShieldConfig(min_green_s=10.0, max_green_s=60.0),
            )
        )

    def test_the_budget_ends_the_run_cleanly_with_a_resumable_checkpoint(
        self, tmp_path: Path
    ) -> None:
        config = TrainingRunConfig(
            run_name="budget",
            num_iterations=50,  # far more than can fit...
            num_steps=4,
            episode_decisions=100,
            checkpoint_dir=tmp_path,
            max_wall_s=0.001,  # ...in a one-millisecond budget
        )

        net = run_training(self._env, config)

        ckpt = checkpoint_path(tmp_path, "budget")
        assert ckpt.is_file()
        fresh = ActorCritic(OBS_DIM, ACTION_DIM, 2)
        optimizer = torch.optim.Adam(fresh.parameters())
        from atlas.marl.training import RunningRewardNormalizer

        last_completed = load_checkpoint(ckpt, fresh, optimizer, RunningRewardNormalizer())
        assert last_completed == 0  # stopped after the very first iteration, not iteration 49
        assert isinstance(net, ActorCritic)

    def test_the_budget_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            TrainingRunConfig(
                run_name="x",
                num_iterations=1,
                num_steps=1,
                episode_decisions=1,
                checkpoint_dir=Path("."),
                max_wall_s=0,
            )
