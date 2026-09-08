"""Validates select_action_for_mode against real junction data (sandbox network's B1) --
MaxPressureProgram.from_net_file only parses the built XML, so this needs no running
SUMO/libsumo twin at all.
"""

from __future__ import annotations

import pytest
import torch

from atlas.common.config import project_root
from atlas.control.fixed_time import native_green_durations_from_net_file
from atlas.control.max_pressure import MaxPressureProgram
from atlas.marl.network import ActorCritic
from atlas.services.decision import actor_name_for_mode, select_action_for_mode

NET_FILE = project_root() / "data" / "networks" / "sandbox" / "net.net.xml"
TLS_ID = "B1"


def _program() -> MaxPressureProgram:
    return MaxPressureProgram.from_net_file(TLS_ID, NET_FILE)


class TestActorNameForMode:
    def test_maps_every_mode_to_a_valid_control_commands_actor(self) -> None:
        valid_actors = {
            "rl_agent",
            "max_pressure",
            "fixed_time",
            "actuated",
            "preemption",
            "operator",
        }
        for mode in ("rl", "max_pressure", "fixed_time", "actuated", "off"):
            assert actor_name_for_mode(mode) in valid_actors  # type: ignore[arg-type]


class TestOffMode:
    def test_returns_last_action_unchanged(self) -> None:
        program = _program()
        action = select_action_for_mode("off", program, last_action=1)
        assert action == 1


class TestMaxPressureMode:
    def test_selects_the_phase_with_more_queued_pressure(self) -> None:
        program = _program()
        # All queue on the first green phase's movements, none on the second's --
        # max-pressure must pick phase 0's action index.
        phase_0_movements = program.enabled_movements(program.green_phase_indices[0])
        queue_lengths = {m.from_lane: 10 for m in phase_0_movements}

        action = select_action_for_mode(
            "max_pressure", program, queue_lengths=queue_lengths, last_action=0
        )

        assert program.green_phase_indices[action] == program.green_phase_indices[0]

    def test_raises_without_queue_lengths(self) -> None:
        program = _program()
        with pytest.raises(ValueError, match="queue_lengths"):
            select_action_for_mode("max_pressure", program, last_action=0)


class TestFixedTimeMode:
    def test_follows_the_native_cycle_deterministically(self) -> None:
        program = _program()
        native_durations = native_green_durations_from_net_file(TLS_ID, NET_FILE)
        green_phase_indices, durations_s = native_durations

        action_at_start = select_action_for_mode(
            "fixed_time", program, native_durations=native_durations, sim_time=0.0, last_action=0
        )
        assert program.green_phase_indices[action_at_start] == green_phase_indices[0]

        # Just past phase 0's duration -> must have advanced to phase 1.
        action_after_first_phase = select_action_for_mode(
            "fixed_time",
            program,
            native_durations=native_durations,
            sim_time=durations_s[0] + 0.1,
            last_action=0,
        )
        assert program.green_phase_indices[action_after_first_phase] == green_phase_indices[1]


class TestActuatedMode:
    def test_extends_the_current_phase_when_a_vehicle_is_present(self) -> None:
        program = _program()
        phase_0_movements = program.enabled_movements(program.green_phase_indices[0])
        queue_lengths = {m.from_lane: 1 for m in phase_0_movements}

        action = select_action_for_mode(
            "actuated",
            program,
            queue_lengths=queue_lengths,
            elapsed_in_phase_s=15.0,
            last_action=0,
        )

        assert program.green_phase_indices[action] == program.green_phase_indices[0]

    def test_advances_to_the_next_phase_on_a_gap(self) -> None:
        program = _program()

        action = select_action_for_mode(
            "actuated",
            program,
            queue_lengths={},  # no vehicles anywhere -- a real gap
            elapsed_in_phase_s=15.0,
            last_action=0,
        )

        assert program.green_phase_indices[action] == program.green_phase_indices[1]

    def test_raises_without_queue_lengths_or_elapsed_in_phase_s(self) -> None:
        program = _program()
        with pytest.raises(ValueError, match="elapsed_in_phase_s"):
            select_action_for_mode("actuated", program, queue_lengths={}, last_action=0)


class TestRlMode:
    def test_uses_the_network_to_pick_a_legal_action(self) -> None:
        program = _program()
        num_phases = len(program.green_phase_indices)
        obs_dim = 8
        net = ActorCritic(obs_dim=obs_dim, action_dim=num_phases, num_agents=1)
        obs = torch.zeros(obs_dim)

        action = select_action_for_mode("rl", program, net=net, obs=obs, last_action=0)

        assert 0 <= action < num_phases

    def test_raises_without_net_or_obs(self) -> None:
        program = _program()
        with pytest.raises(ValueError, match="net and obs"):
            select_action_for_mode("rl", program, last_action=0)
