"""Local simulation tiles: finding where an incident is, cutting the network and demand around it,
caching, and running the what-if on that cut in a separate process.

The unit tests use tiny hand-built files; the build/what-if tests use the REAL Midtown network and
demand (SUMO's own netconvert and cutRoutes), so they prove the cut is a valid, runnable simulation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from atlas.cognitive.playbooks import PlaybookProposal
from atlas.cognitive.what_if import WhatIfResult, run_what_if
from atlas.common.config import project_root
from atlas.twin.tiles import TileStore, anchor_points

NETWORK_DIR = project_root() / "data" / "networks" / "manhattan_midtown"
NET_FILE = NETWORK_DIR / "net.net.xml"
ROUTE_FILE = NETWORK_DIR / "demand" / "pm_peak.rou.xml"
VTYPES_FILE = NETWORK_DIR / "vtypes.add.xml"

requires_network = pytest.mark.skipif(
    not (NET_FILE.is_file() and ROUTE_FILE.is_file()), reason="manhattan_midtown not built"
)

TINY_NET = """<net>
  <junction id="J1" type="traffic_light" x="100.0" y="200.0" incLanes="" intLanes="" shape=""/>
  <edge id=":J1_0" function="internal"><lane id=":J1_0_0" shape="1,1 2,2"/></edge>
  <edge id="E1" from="J0" to="J1"><lane id="E1_0" shape="10.5,20.5 90.0,190.0"/></edge>
  <edge id="E2" from="J1" to="J2"><lane id="E2_0" shape="110.0,210.0 300.0,400.0"/></edge>
</net>"""


@pytest.fixture
def tiny_net(tmp_path: Path) -> Path:
    path = tmp_path / "tiny.net.xml"
    path.write_text(TINY_NET, encoding="utf-8")
    return path


class TestAnchorPoints:
    def test_a_road_is_anchored_at_its_first_point_and_a_junction_at_its_centre(
        self, tiny_net: Path
    ) -> None:
        points = anchor_points(tiny_net, ["E1", "J1"])

        assert points == {"E1": (10.5, 20.5), "J1": (100.0, 200.0)}

    def test_unknown_ids_are_simply_absent(self, tiny_net: Path) -> None:
        assert anchor_points(tiny_net, ["nope"]) == {}


class TestTileKeys:
    def _store(self, tmp_path: Path) -> TileStore:
        return TileStore(Path("n"), Path("r"), tmp_path, grid_m=300.0)

    def test_nearby_incidents_share_a_tile(self, tmp_path: Path) -> None:
        store = self._store(tmp_path)

        assert store.key_for((610.0, 905.0))[0] == store.key_for((590.0, 880.0))[0]

    def test_far_apart_incidents_get_different_tiles(self, tmp_path: Path) -> None:
        store = self._store(tmp_path)

        assert store.key_for((0.0, 0.0))[0] != store.key_for((3000.0, 0.0))[0]

    def test_a_tile_is_centred_on_its_grid_cell(self, tmp_path: Path) -> None:
        assert self._store(tmp_path).key_for((620.0, 10.0))[1] == (600.0, 0.0)

    def test_links_that_do_not_exist_have_no_tile(self, tiny_net: Path, tmp_path: Path) -> None:
        store = TileStore(tiny_net, Path("r"), tmp_path)

        assert store.for_links(["ghost"]) is None


@requires_network
class TestBuildingATile:
    def test_a_built_tile_is_a_smaller_valid_network_with_the_traffic_that_crosses_it(
        self, tmp_path: Path
    ) -> None:
        # Radius larger than the district so the cut is just "all of it": proves the pipeline
        # (netconvert cut, cutRoutes) produces a real, loadable network and demand.
        store = TileStore(NET_FILE, ROUTE_FILE, tmp_path, radius_m=5000.0)

        tile = store.for_links(["42430342"])

        assert tile is not None
        assert tile.net_file.is_file() and tile.route_file.is_file()
        assert "<vehicle " in tile.route_file.read_text(encoding="utf-8")
        from atlas.twin.incident_effects import RoadIndex

        assert "1117867028" in RoadIndex.from_net_file(tile.net_file).lane_counts

    def test_a_second_request_for_the_same_area_reuses_the_cached_tile(
        self, tmp_path: Path
    ) -> None:
        store = TileStore(NET_FILE, ROUTE_FILE, tmp_path, radius_m=5000.0)
        store.for_links(["42430342"])

        with patch.object(TileStore, "_build", side_effect=AssertionError("rebuilt")):
            tile = store.for_links(["42430342"])

        assert tile is not None

    def test_a_half_built_tile_never_looks_finished(self, tmp_path: Path) -> None:
        store = TileStore(NET_FILE, ROUTE_FILE, tmp_path, radius_m=5000.0)

        with (
            patch("atlas.twin.tiles.subprocess.run", side_effect=RuntimeError("boom")),
            pytest.raises(RuntimeError),
        ):
            store.for_links(["42430342"])

        assert not list(tmp_path.glob("*/net.net.xml"))


class TestTileWhatIfFailsSafe:
    """Any problem is `checked=False` (no evidence), never a guessed answer."""

    def _proposal(self, name: str = "divert_arterial", **params: Any) -> PlaybookProposal:
        return PlaybookProposal(playbook_name=name, params=params, rationale="t")

    def test_no_tile_for_the_location_means_unchecked(self) -> None:
        class NoTile:
            def for_links(self, links: Any) -> None:
                return None

        result = run_what_if(
            self._proposal(from_edge="a"),
            Path("n"),
            Path("r"),
            incident={"class": "accident", "location": {"link_ids": ["a"]}},
            tiles=NoTile(),  # type: ignore[arg-type]
        )

        assert result.checked is False

    def test_no_location_means_unchecked_without_building_anything(self) -> None:
        class Boom:
            def for_links(self, links: Any) -> None:
                raise AssertionError("must not build a tile for no location")

        result = run_what_if(
            self._proposal(from_edge="a"),
            Path("n"),
            Path("r"),
            incident={"class": "accident", "location": {}},
            tiles=Boom(),  # type: ignore[arg-type]
        )

        assert result.checked is False

    def test_a_failing_worker_means_unchecked(self, tmp_path: Path) -> None:
        class FakeTile:
            key = "k"
            net_file = tmp_path / "n"
            route_file = tmp_path / "r"

        class Store:
            def for_links(self, links: Any) -> FakeTile:
                return FakeTile()

        crashed = type("P", (), {"returncode": 1, "stdout": "", "stderr": "boom"})()
        with patch("atlas.cognitive.what_if.subprocess.run", return_value=crashed):
            result = run_what_if(
                self._proposal(from_edge="a"),
                Path("n"),
                Path("r"),
                incident={"class": "accident", "location": {"link_ids": ["a"]}},
                tiles=Store(),  # type: ignore[arg-type]
            )

        assert result.checked is False

    def test_the_workers_json_line_becomes_the_result(self, tmp_path: Path) -> None:
        class FakeTile:
            key = "k"
            net_file = tmp_path / "n"
            route_file = tmp_path / "r"

        class Store:
            def for_links(self, links: Any) -> FakeTile:
                return FakeTile()

        line = (
            '{"checked": true, "improved": true, "baseline_mean_pressure": 10.0, '
            '"proposed_mean_pressure": 8.0, "metric": "mean_halted_vehicles"}'
        )
        done = type("P", (), {"returncode": 0, "stdout": f"noise\n{line}\n", "stderr": ""})()
        with patch("atlas.cognitive.what_if.subprocess.run", return_value=done):
            result = run_what_if(
                self._proposal(from_edge="a"),
                Path("n"),
                Path("r"),
                incident={"class": "accident", "location": {"link_ids": ["a"]}},
                tiles=Store(),  # type: ignore[arg-type]
            )

        assert result == WhatIfResult(True, True, 10.0, 8.0, "mean_halted_vehicles")


@requires_network
class TestEndToEndOnARealTile:
    def test_a_real_what_if_runs_in_a_worker_on_a_cut_of_the_network(self, tmp_path: Path) -> None:
        store = TileStore(NET_FILE, ROUTE_FILE, tmp_path, radius_m=5000.0)

        result = run_what_if(
            PlaybookProposal(
                playbook_name="divert_arterial",
                params={"from_edge": "1117867028"},
                rationale="t",
            ),
            NET_FILE,
            ROUTE_FILE,
            additional_files=[VTYPES_FILE],
            incident={
                "incident_id": "tile-e2e",
                "class": "accident",
                "location": {"link_ids": ["42430342"]},
            },
            tiles=store,
        )

        assert result.checked is True
        assert result.metric == "mean_halted_vehicles"
        assert result.baseline_mean_pressure is not None
        assert result.proposed_mean_pressure is not None
