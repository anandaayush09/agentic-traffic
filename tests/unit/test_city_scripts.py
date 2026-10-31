"""Validates the city-scale tooling: the network audit (against a tiny hand-built network whose
right answers are known, including signals it must SKIP and say why), plus the pure parts of the
fetch and build scripts. The real Manhattan files are far too big for unit tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import audit_network
import build_network
import fetch_city_osm
from atlas.common.config import project_root

TINY_NET = """<?xml version="1.0" encoding="UTF-8"?>
<net>
  <edge id=":j1_0" function="internal"><lane id=":j1_0_0" length="5"/></edge>
  <edge id="a" from="n0" to="j1"><lane id="a_0" length="100"/><lane id="a_1" length="100"/></edge>
  <edge id="b" from="n0" to="j2"><lane id="b_0" length="50"/></edge>
  <edge id="c" from="j1" to="n9"><lane id="c_0" length="25"/></edge>
  <junction id="j1" type="traffic_light"/>
  <junction id="j2" type="traffic_light"/>
  <junction id="j3" type="traffic_light"/>
  <junction id="j4" type="traffic_light"/>
  <junction id="n0" type="dead_end"/>

  <!-- j1: two green phases + yellow transitions: controllable -->
  <tlLogic id="j1" type="static" programID="0" offset="0">
    <phase duration="30" state="GGrr"/>
    <phase duration="3" state="yyrr"/>
    <phase duration="30" state="rrGG"/>
    <phase duration="3" state="rryy"/>
  </tlLogic>
  <!-- j2: a single green phase: controllable -->
  <tlLogic id="j2" type="static" programID="0" offset="0">
    <phase duration="30" state="GG"/>
    <phase duration="3" state="yy"/>
  </tlLogic>
  <!-- j3: has a program and connections but never a green: must be skipped -->
  <tlLogic id="j3" type="static" programID="0" offset="0">
    <phase duration="10" state="rr"/>
  </tlLogic>
  <!-- j4: a program but no controlled connections: must be skipped -->
  <tlLogic id="j4" type="static" programID="0" offset="0">
    <phase duration="30" state="GG"/>
  </tlLogic>

  <connection from="a" to="c" fromLane="0" toLane="0" tl="j1" linkIndex="0"/>
  <connection from="a" to="c" fromLane="1" toLane="0" tl="j1" linkIndex="1"/>
  <connection from="b" to="c" fromLane="0" toLane="0" tl="j2" linkIndex="0"/>
  <connection from="a" to="c" fromLane="0" toLane="0" tl="j3" linkIndex="0"/>
  <connection from="c" to="a" fromLane="0" toLane="1"/>
</net>
"""


@pytest.fixture
def tiny_net(tmp_path: Path) -> Path:
    path = tmp_path / "net.net.xml"
    path.write_text(TINY_NET, encoding="utf-8")
    return path


class TestAudit:
    def test_counts_real_roads_but_not_internal_junction_edges(self, tiny_net: Path) -> None:
        report = audit_network.audit(tiny_net)

        assert report["roads"] == 3  # a, b, c -- not ":j1_0"
        assert report["lanes"] == 4
        assert report["lane_km"] == pytest.approx(0.275)  # (100+100+50+25) m
        assert report["roads_by_lane_count"] == {1: 2, 2: 1}

    def test_separates_controllable_signals_from_skipped_ones_and_says_why(
        self, tiny_net: Path
    ) -> None:
        report = audit_network.audit(tiny_net)

        assert report["signals_total"] == 4
        assert report["controllable_ids"] == ["j1", "j2"]
        assert report["signals_skipped"] == {
            "j3": "no controllable green phase",
            "j4": "no controlled connections",
        }

    def test_reports_the_sizes_the_control_stack_must_be_built_for(self, tiny_net: Path) -> None:
        report = audit_network.audit(tiny_net)

        # j1 has 2 incoming lanes and 2 green phases; j2 has 1 lane and 1 green phase.
        assert report["incoming_lanes_per_signal"] == {1: 1, 2: 1}
        assert report["green_phases_per_signal"] == {1: 1, 2: 1}
        assert report["required_max_lanes"] == 2
        assert report["required_max_green_phases"] == 2
        assert report["signals_over_district_limits"] == 0

    def test_flags_signals_beyond_the_district_limits(self, tmp_path: Path) -> None:
        lanes = "".join(f'<lane id="e_{i}" length="10"/>' for i in range(12))
        conns = "".join(
            f'<connection from="e" to="f" fromLane="{i}" toLane="0" tl="big" linkIndex="{i}"/>'
            for i in range(12)
        )
        net = tmp_path / "big.net.xml"
        net.write_text(
            f"""<net><edge id="e">{lanes}</edge><junction id="big" type="traffic_light"/>
            <tlLogic id="big" type="static"><phase duration="9" state="{"G" * 12}"/></tlLogic>
            {conns}</net>""",
            encoding="utf-8",
        )

        report = audit_network.audit(net)

        assert report["required_max_lanes"] == 12
        assert report["signals_over_district_limits"] == 1

    def test_the_real_midtown_district_matches_its_known_facts(self) -> None:
        net = project_root() / "data" / "networks" / "manhattan_midtown" / "net.net.xml"
        if not net.is_file():
            pytest.skip("manhattan_midtown network not built")

        report = audit_network.audit(net)

        assert report["signals_total"] == 22
        assert report["signals_controllable"] == 22
        assert report["required_max_lanes"] == 10  # the value the district code hard-codes
        assert report["required_max_green_phases"] == 2

    def test_the_report_renders_for_a_human(self, tiny_net: Path) -> None:
        text = audit_network.render(audit_network.audit(tiny_net))

        assert "2 controllable by ATLAS" in text
        assert "skipped signals by reason" in text
        assert "max_lanes >= 2" in text


class TestFetchScript:
    def test_the_query_targets_the_borough_by_relation_and_only_drivable_roads(self) -> None:
        query = fetch_city_osm.build_query(8398124)

        assert "rel(8398124);map_to_area" in query
        assert "motorway" in query and "residential" in query
        assert "service" not in query  # driveways/parking aisles would dwarf the real network
        assert "footway" not in query and "cycleway" not in query

    def test_every_known_city_has_a_relation_id(self) -> None:
        for spec in fetch_city_osm.CITIES.values():
            assert isinstance(spec["relation"], int)
            assert spec["label"]


class TestBuildScript:
    def test_the_city_build_reuses_the_districts_proven_flags_and_keeps_only_car_roads(
        self,
    ) -> None:
        from build_district_network import NETCONVERT_FLAGS

        cmd = build_network.build_command(Path("raw.osm.xml"), Path("net.net.xml"), Path("t.typ"))

        assert all(flag in cmd for flag in NETCONVERT_FLAGS)
        assert cmd[cmd.index("--keep-edges.by-vclass") + 1] == "passenger"
        assert cmd[cmd.index("--osm-files") + 1] == "raw.osm.xml"

    def test_building_without_an_extract_explains_how_to_get_one(self) -> None:
        with pytest.raises(SystemExit, match="fetch_city_osm.py --city no_such_city"):
            build_network.build("no_such_city")
