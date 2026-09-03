"""Validates max-pressure against a real junction (42430333) from the built network,
using hand-computed pressure values rather than only checking "which phase wins" -- per
the Week 3 roadmap's "tested against hand-computed cases".

Movement structure for this junction (verified against net.net.xml before writing this
test -- see src/atlas/control/max_pressure.py's module docstring):

Phase 0 "GGGgrrrrrrGGgg" (green at linkIndex 0,1,2,3,10,11,12,13) enables movements from
lanes: 1320232776#0_0 (x2), 1320232776#0_1, 1320232776#0_2, -167922072#1_0,
-167922072#1_1 (x3).

Phase 2 "rrrrGGGGGGrrrr" (green at linkIndex 4,5,6,7,8,9) enables movements from lanes:
448686048#0_0 (x2), 448686048#0_1, 448686048#0_2, 448686048#0_3, 448686048#0_4.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from atlas.common.config import project_root
from atlas.control.max_pressure import (
    MaxPressureProgram,
    phase_pressure,
    select_max_pressure_phase,
)

TLS_ID = "42430333"


@pytest.fixture(scope="session")
def net_file() -> Path:
    import sys

    sys.path.insert(0, str(project_root() / "scripts"))
    import build_district_network

    build_district_network.build_network()
    return build_district_network.NET_FILE


@pytest.fixture(scope="session")
def program(net_file: Path) -> MaxPressureProgram:
    return MaxPressureProgram.from_net_file(TLS_ID, net_file)


class TestMaxPressureProgramParsing:
    def test_finds_both_green_phases(self, program: MaxPressureProgram) -> None:
        assert program.green_phase_indices == (0, 2)

    def test_phase_0_enables_eight_movements(self, program: MaxPressureProgram) -> None:
        movements = program.enabled_movements(0)

        assert len(movements) == 8
        assert {m.link_index for m in movements} == {0, 1, 2, 3, 10, 11, 12, 13}

    def test_phase_2_enables_six_movements(self, program: MaxPressureProgram) -> None:
        movements = program.enabled_movements(2)

        assert len(movements) == 6
        assert {m.link_index for m in movements} == {4, 5, 6, 7, 8, 9}

    def test_a_lane_can_control_multiple_movements(self, program: MaxPressureProgram) -> None:
        # 1320232776#0_0 is the from_lane of both linkIndex 0 (right turn) and
        # linkIndex 1 (straight) -- confirmed against the real network before writing
        # this; pressure must count both, not deduplicate by lane (see module docstring).
        phase_0_from_lanes = [m.from_lane for m in program.enabled_movements(0)]

        assert phase_0_from_lanes.count("1320232776#0_0") == 2


class TestPhasePressureHandComputed:
    def test_phase_0_pressure_with_only_its_upstream_queues_loaded(
        self, program: MaxPressureProgram
    ) -> None:
        queue_lengths = {
            "1320232776#0_0": 10,
            "1320232776#0_1": 10,
            "1320232776#0_2": 10,
            "-167922072#1_0": 10,
            "-167922072#1_1": 10,
        }
        # Phase 0 has 8 movements; four of them share from_lane -167922072#1_1 (x3) and
        # 1320232776#0_0 (x2) -- 5 distinct from_lanes above cover all 8 movements,
        # each contributing +10 (all to_lanes are 0/absent), total 8 * 10 = 80.

        assert phase_pressure(program, 0, queue_lengths) == 80
        assert phase_pressure(program, 2, queue_lengths) == 0

    def test_phase_2_pressure_with_only_its_upstream_queues_loaded(
        self, program: MaxPressureProgram
    ) -> None:
        queue_lengths = {
            "448686048#0_0": 10,
            "448686048#0_1": 10,
            "448686048#0_2": 10,
            "448686048#0_3": 10,
            "448686048#0_4": 10,
        }
        # Phase 2 has 6 movements from 5 distinct from_lanes (448686048#0_0 covers two:
        # linkIndex 4 and 5), each contributing +10, total 6 * 10 = 60.

        assert phase_pressure(program, 2, queue_lengths) == 60
        assert phase_pressure(program, 0, queue_lengths) == 0

    def test_downstream_congestion_reduces_pressure(self, program: MaxPressureProgram) -> None:
        # linkIndex 10 (-167922072#1_0 -> -1320232776#1_0): upstream queue 10, downstream
        # queue 4 -> pressure for this one movement is 10 - 4 = 6, not 10.
        queue_lengths = {"-167922072#1_0": 10, "-1320232776#1_0": 4}

        pressure = phase_pressure(program, 0, queue_lengths)

        assert pressure == 6  # only this one movement is non-zero


class TestSelectMaxPressurePhase:
    def test_selects_phase_with_higher_pressure(self, program: MaxPressureProgram) -> None:
        queue_lengths = {"1320232776#0_0": 20}  # only feeds phase 0's movements

        assert select_max_pressure_phase(program, queue_lengths) == 0

    def test_selects_the_other_phase_when_it_dominates(self, program: MaxPressureProgram) -> None:
        queue_lengths = {"448686048#0_0": 20}  # only feeds phase 2's movements

        assert select_max_pressure_phase(program, queue_lengths) == 2

    def test_ties_break_to_lowest_phase_index(self, program: MaxPressureProgram) -> None:
        assert select_max_pressure_phase(program, {}) == program.green_phase_indices[0]
