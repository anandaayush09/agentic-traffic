"""Validates the closed-loop controlled twin against the real Manhattan network + pm_peak
demand: controller modes really drive the junctions, and a compiled playbook effect
really changes what happens on the road and reverts on schedule.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from atlas.common.config import project_root
from atlas.twin.controlled_twin import (
    CONTROLLER_MODES,
    CORRIDOR_TLS_IDS,
    DISTRICT_TLS_IDS,
    ControlledTwin,
    edge_halted_total,
)
from atlas.twin.incident_effects import RoadIndex
from atlas.twin.playbook_effects import PlaybookEffect, compile_playbook

NETWORK_DIR = project_root() / "data" / "networks" / "manhattan_midtown"
NET_FILE = NETWORK_DIR / "net.net.xml"
ROUTE_FILE = NETWORK_DIR / "demand" / "pm_peak.rou.xml"
VTYPES_FILE = NETWORK_DIR / "vtypes.add.xml"

pytestmark = pytest.mark.skipif(
    not (NET_FILE.is_file() and ROUTE_FILE.is_file()),
    reason="manhattan_midtown network not built",
)

JUNCTION = "42430342"


@pytest.fixture(scope="module")
def index() -> RoadIndex:
    return RoadIndex.from_net_file(NET_FILE)


def _twin(index: RoadIndex) -> ControlledTwin:
    return ControlledTwin(NET_FILE, ROUTE_FILE, [VTYPES_FILE], road_index=index)


@pytest.fixture
def twin(index: RoadIndex) -> Iterator[ControlledTwin]:
    t = _twin(index)
    t.reset(seed=42)
    t.step_decisions(12)  # 60 s: some traffic on the roads
    try:
        yield t
    finally:
        t.close()


class TestModes:
    def test_every_corridor_junction_starts_on_the_default_mode(self, twin: ControlledTwin) -> None:
        states = twin.junction_states()

        assert set(states) == set(DISTRICT_TLS_IDS)
        assert {s["mode"] for s in states.values()} == {"fixed_time"}
        assert not any(s["playbook_override"] for s in states.values())

    def test_set_mode_changes_the_effective_mode(self, twin: ControlledTwin) -> None:
        twin.set_mode(JUNCTION, "max_pressure")

        assert twin.junction_states()[JUNCTION]["mode"] == "max_pressure"
        twin.step_decisions(2)  # and the loop keeps running on it without error

    def test_every_live_mode_runs_a_decision_loop(self, twin: ControlledTwin) -> None:
        for mode in CONTROLLER_MODES:
            twin.set_mode(JUNCTION, mode)
            twin.step_decisions(2)

    def test_rl_is_not_available_live(self, twin: ControlledTwin) -> None:
        with pytest.raises(ValueError, match="mode must be one of"):
            twin.set_mode(JUNCTION, "rl")

    def test_uncontrolled_junction_is_rejected(self, twin: ControlledTwin) -> None:
        with pytest.raises(KeyError, match="not an ATLAS-controlled junction"):
            twin.set_mode("not-a-junction", "max_pressure")


class TestPlaybookEffects:
    def test_mode_override_wins_then_reverts_when_it_expires(
        self, twin: ControlledTwin, index: RoadIndex
    ) -> None:
        effect = compile_playbook(
            "switch_to_max_pressure",
            {"junction_ids": [JUNCTION]},
            index,
            CORRIDOR_TLS_IDS,
            default_duration_s=20.0,
        )
        assert effect is not None
        twin.apply_effect("e1", effect)

        during = twin.junction_states()[JUNCTION]
        assert (during["mode"], during["playbook_override"]) == ("max_pressure", True)

        twin.step_decisions(6)  # 30 s > the 20 s effect
        after = twin.junction_states()[JUNCTION]
        assert (after["mode"], after["playbook_override"]) == ("fixed_time", False)
        assert twin.active_effects() == []

    def test_an_effect_id_cannot_be_applied_twice(
        self, twin: ControlledTwin, index: RoadIndex
    ) -> None:
        effect = PlaybookEffect("x", "x", {}, (), 60.0, ())
        twin.apply_effect("dup", effect)

        with pytest.raises(ValueError, match="already active"):
            twin.apply_effect("dup", effect)

    def test_divert_really_empties_the_penalised_roads(self, index: RoadIndex) -> None:
        edges = index.resolve([JUNCTION])

        def mean_on_edges(divert: bool) -> float:
            twin = _twin(index)
            twin.reset(seed=42)
            try:
                twin.step_decisions(60)  # 300 s warm-up
                if divert:
                    effect = compile_playbook(
                        "divert_arterial",
                        {"from_edge": JUNCTION, "to_edge": JUNCTION, "duration_s": 300.0},
                        index,
                        CORRIDOR_TLS_IDS,
                    )
                    assert effect is not None
                    twin.apply_effect("d1", effect)
                conn = twin.connection
                samples = []
                for _ in range(48):  # 240 s, inside the effect window
                    twin.step_decisions(1)
                    samples.append(sum(int(conn.edge.getLastStepVehicleNumber(e)) for e in edges))
                return sum(samples) / len(samples)
            finally:
                twin.close()

        assert mean_on_edges(divert=True) < mean_on_edges(divert=False)

    def test_avoidance_is_released_when_the_effect_expires(
        self, twin: ControlledTwin, index: RoadIndex
    ) -> None:
        effect = compile_playbook(
            "divert_arterial",
            {"from_edge": JUNCTION, "to_edge": JUNCTION, "duration_s": 10.0},
            index,
            CORRIDOR_TLS_IDS,
        )
        assert effect is not None
        twin.apply_effect("d2", effect)

        twin.step_decisions(4)  # 20 s > 10 s

        assert twin.active_effects() == []
        assert not any(count > 0 for count in twin._avoid_counts.values())
        # and a later decision with nothing penalised still runs
        twin.step_decisions(2)
        assert edge_halted_total(twin.connection, index.lane_counts) >= 0
