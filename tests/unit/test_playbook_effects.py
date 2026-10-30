"""Validates the playbook-to-twin-effect compiler and the incident road resolution it
builds on. Pure logic over a tiny hand-built road network (A -> B -> C along one line:
n1 --A--> n2 --B--> n3 --C--> n4), so each rule is checked in isolation; the real-network
behaviour lives in test_controlled_twin.py.
"""

from __future__ import annotations

from atlas.twin.incident_effects import RoadIndex, link_ids_of
from atlas.twin.playbook_effects import COMPILABLE_PLAYBOOKS, compile_playbook

INDEX = RoadIndex(
    lane_counts={"A": 2, "B": 2, "C": 1},
    free_flow_mps={"A": 13.9, "B": 13.9, "C": 13.9},
    length_m={"A": 100.0, "B": 100.0, "C": 100.0},
    from_node={"A": "n1", "B": "n2", "C": "n3"},
    to_node={"A": "n2", "B": "n3", "C": "n4"},
    incoming={"n1": (), "n2": ("A",), "n3": ("B",), "n4": ("C",)},
)


class TestRoadIndexResolve:
    def test_road_id_resolves_to_itself(self) -> None:
        assert INDEX.resolve(["B"]) == ["B"]

    def test_junction_id_resolves_to_the_roads_entering_it(self) -> None:
        assert INDEX.resolve(["n3"]) == ["B"]

    def test_unknown_ids_are_skipped_and_duplicates_dropped(self) -> None:
        assert INDEX.resolve(["nowhere", "B", "n3", "B"]) == ["B"]

    def test_link_ids_of_tolerates_missing_or_malformed_locations(self) -> None:
        assert link_ids_of({"link_ids": ["x", 2]}) == ["x", "2"]
        assert link_ids_of({"link_ids": "x"}) == []
        assert link_ids_of(None) == []


class TestHoldCurrentPlan:
    def test_is_an_explicit_no_op(self) -> None:
        effect = compile_playbook("hold_current_plan", {"reason": "r"}, INDEX, ["n2"])

        assert effect is not None
        assert effect.is_noop


class TestSwitchToMaxPressure:
    def test_forces_max_pressure_on_the_controlled_junctions_only(self) -> None:
        effect = compile_playbook(
            "switch_to_max_pressure", {"junction_ids": ["n2", "n9"]}, INDEX, ["n2", "n3"]
        )

        assert effect is not None
        assert dict(effect.junction_modes) == {"n2": "max_pressure"}
        assert effect.affected_edges == ("A",)

    def test_none_of_the_junctions_controlled_cannot_compile(self) -> None:
        assert (
            compile_playbook("switch_to_max_pressure", {"junction_ids": ["n9"]}, INDEX, ["n2"])
            is None
        )

    def test_malformed_params_cannot_compile(self) -> None:
        assert (
            compile_playbook("switch_to_max_pressure", {"junction_ids": []}, INDEX, ["n2"]) is None
        )
        assert compile_playbook("switch_to_max_pressure", {}, INDEX, ["n2"]) is None


class TestDivertArterial:
    def test_penalises_the_named_road(self) -> None:
        effect = compile_playbook(
            "divert_arterial",
            {"from_edge": "B", "to_edge": "B", "duration_s": 600.0},
            INDEX,
            [],
        )

        assert effect is not None
        assert effect.avoid_edges == ("B",)
        assert effect.affected_edges == ("B",)
        assert effect.duration_s == 600.0
        assert not effect.junction_modes

    def test_a_junction_id_means_every_road_entering_it(self) -> None:
        effect = compile_playbook(
            "divert_arterial", {"from_edge": "n3", "to_edge": "n3", "duration_s": 60.0}, INDEX, []
        )

        assert effect is not None
        assert effect.avoid_edges == ("B",)

    def test_unknown_road_cannot_compile(self) -> None:
        assert (
            compile_playbook(
                "divert_arterial",
                {"from_edge": "zzz", "to_edge": "zzz", "duration_s": 60},
                INDEX,
                [],
            )
            is None
        )


class TestMeterUpstream:
    def test_runs_max_pressure_on_controlled_junctions_upstream_of_the_road(self) -> None:
        # C starts at n3; one hop further upstream is n2 (B starts there). Only n2 is
        # controlled, so only n2 is metered.
        effect = compile_playbook(
            "meter_upstream", {"corridor_id": "C", "rate": 0.7}, INDEX, ["n2"]
        )

        assert effect is not None
        assert dict(effect.junction_modes) == {"n2": "max_pressure"}
        assert effect.affected_edges == ("C",)

    def test_no_controlled_junction_upstream_cannot_compile(self) -> None:
        assert (
            compile_playbook("meter_upstream", {"corridor_id": "C", "rate": 0.7}, INDEX, ["n1"])
            is None
        )


class TestUnsupportedPlaybooks:
    def test_weather_derate_and_bus_bridge_have_no_defensible_simulation(self) -> None:
        assert (
            compile_playbook(
                "weather_capacity_derate", {"area_id": "A", "factor": 0.6}, INDEX, ["n2"]
            )
            is None
        )
        assert compile_playbook("bus_bridge", {"route_id": "r"}, INDEX, ["n2"]) is None

    def test_compilable_set_matches_what_actually_compiles(self) -> None:
        assert {
            "switch_to_max_pressure",
            "divert_arterial",
            "meter_upstream",
            "hold_current_plan",
        } == COMPILABLE_PLAYBOOKS
