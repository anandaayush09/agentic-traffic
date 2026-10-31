"""Local simulation tiles: a small cut of a big network, with the demand that actually crosses it.

Why: the what-if tool simulates an incident with and without a response. On a whole borough that is
two full simulations per incident (a 40 MB network, thousands of signals) -- minutes, and more RAM
than is sensible inside a request. The response to an incident is local, so the simulation can be
too: take the roads within `radius_m` of the incident, keep the traffic that really passes through
that window, and simulate only that. A tile is a few hundred roads and simulates in seconds.

How a tile is made (all with SUMO's own tools, nothing hand-rolled):
1. `netconvert --keep-edges.in-boundary` cuts the network (road ids are preserved, so the incident's
   road ids and the playbook's junction ids mean the same thing in the tile);
2. `cutRoutes.py` clips every vehicle's route to the tile and re-times its departure to when it
   enters, so the tile sees the traffic the full run would send through it.

Tiles are snapped to a grid (`grid_m`), so nearby incidents share one, and cached on disk: the first
incident in an area pays the ~15 s to build it, later ones reuse it. `prepare()` builds tiles ahead of
time (e.g. overnight, or for every junction).

What a tile result is NOT: it ignores what happens outside the window (spill-back beyond the edge,
drivers rerouting from far away). It answers "does this response help locally", which is what the
operator is being asked to approve; it is not a citywide forecast.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

from atlas.common.logging import get_logger
from atlas.common.sumo_locate import get_binary_path, get_sumo_home

logger = get_logger(__name__)

DEFAULT_RADIUS_M = 600.0
DEFAULT_GRID_M = 300.0
# cutRoutes counts EDGES here, not metres: a route needs at least 2 roads inside the tile to matter.
MIN_ROUTE_EDGES = 2


@dataclass(frozen=True, slots=True)
class Tile:
    key: str
    net_file: Path
    route_file: Path
    center: tuple[float, float]


def anchor_points(net_file: Path, link_ids: Iterable[str]) -> dict[str, tuple[float, float]]:
    """x,y (network coordinates) of each requested road (its first point) or junction (its centre),
    from one streaming pass over the network file (cheap next to loading it into sumolib)."""
    wanted = set(link_ids)
    found: dict[str, tuple[float, float]] = {}
    for _, element in ElementTree.iterparse(net_file, events=("end",)):
        if element.tag == "junction":
            junction_id = element.get("id", "")
            if junction_id in wanted and element.get("x") is not None:
                found[junction_id] = (float(element.get("x", "0")), float(element.get("y", "0")))
            element.clear()
        elif element.tag == "edge":
            edge_id = element.get("id", "")
            if edge_id in wanted:
                lane = element.find("lane")
                shape = lane.get("shape", "") if lane is not None else ""
                first = shape.split(" ")[0].split(",") if shape else []
                if len(first) >= 2:
                    found[edge_id] = (float(first[0]), float(first[1]))
            element.clear()
            if len(found) == len(wanted):
                break
        elif element.tag in ("connection", "tlLogic"):
            element.clear()
    return found


class TileStore:
    def __init__(
        self,
        net_file: Path,
        route_file: Path,
        cache_dir: Path,
        *,
        radius_m: float = DEFAULT_RADIUS_M,
        grid_m: float = DEFAULT_GRID_M,
    ) -> None:
        self._net_file = net_file
        self._route_file = route_file
        self._cache_dir = cache_dir
        self._radius_m = radius_m
        self._grid_m = grid_m
        self._lock = threading.Lock()

    @property
    def net_file(self) -> Path:
        return self._net_file

    def key_for(self, point: tuple[float, float]) -> tuple[str, tuple[float, float]]:
        """Grid-snapped tile key and centre for a network coordinate."""
        gx = round(point[0] / self._grid_m)
        gy = round(point[1] / self._grid_m)
        return f"{gx}_{gy}", (gx * self._grid_m, gy * self._grid_m)

    def for_links(self, link_ids: Sequence[str]) -> Tile | None:
        """The tile covering these roads/junctions, built (and cached) if needed. None if none of them
        can be located in the network."""
        anchors = anchor_points(self._net_file, link_ids)
        if not anchors:
            return None
        xs = [p[0] for p in anchors.values()]
        ys = [p[1] for p in anchors.values()]
        return self.for_point((sum(xs) / len(xs), sum(ys) / len(ys)))

    def for_point(self, point: tuple[float, float]) -> Tile:
        key, center = self.key_for(point)
        tile = Tile(
            key=key,
            net_file=self._cache_dir / key / "net.net.xml",
            route_file=self._cache_dir / key / "routes.rou.xml",
            center=center,
        )
        with self._lock:  # one build at a time; concurrent requests for the same tile share it
            if not (tile.net_file.is_file() and tile.route_file.is_file()):
                self._build(tile)
        return tile

    def prepare(self, points: Iterable[tuple[float, float]]) -> list[Tile]:
        """Builds the tiles covering these points ahead of time (idempotent)."""
        return [self.for_point(p) for p in points]

    # -- internals --

    def _build(self, tile: Tile) -> None:
        directory = tile.net_file.parent
        work = directory.with_name(directory.name + ".part")
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        cx, cy = tile.center
        r = self._radius_m
        net_out = work / "net.net.xml"
        routes_out = work / "routes.rou.xml"
        logger.info("tiles.building", key=tile.key, radius_m=r)

        netconvert = subprocess.run(  # noqa: S603 -- fixed binary, our own arguments
            [
                str(get_binary_path("netconvert")),
                "-s",
                str(self._net_file),
                "--keep-edges.in-boundary",
                f"{cx - r},{cy - r},{cx + r},{cy + r}",
                "-o",
                str(net_out),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if netconvert.returncode != 0 or not net_out.is_file():
            shutil.rmtree(work, ignore_errors=True)
            raise RuntimeError(f"tile network cut failed: {netconvert.stderr[-300:]}")

        cut = subprocess.run(  # noqa: S603
            [
                sys.executable,
                str(get_sumo_home() / "tools" / "route" / "cutRoutes.py"),
                str(net_out),
                str(self._route_file),
                "--routes-output",
                str(routes_out),
                "--min-length",
                str(MIN_ROUTE_EDGES),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if cut.returncode != 0 or not routes_out.is_file():
            shutil.rmtree(work, ignore_errors=True)
            raise RuntimeError(f"tile route cut failed: {(cut.stdout + cut.stderr)[-300:]}")

        # Publish atomically: a half-built tile directory must never look finished.
        shutil.rmtree(directory, ignore_errors=True)
        work.replace(directory)
        logger.info("tiles.built", key=tile.key)
