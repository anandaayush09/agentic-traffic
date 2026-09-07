#!/usr/bin/env python
"""Exports real lon/lat junction coordinates for the Week 5 corridor (5 contiguous
manhattan_midtown junctions) to a static JSON asset the Week 10 React+MapLibre
dashboard reads directly -- MapLibre needs WGS84 lon/lat, but SUMO junctions are only
stored in the network's own projected UTM coordinates.

Real data, not placeholder pins: reads the actual built network and inverts its own
UTM projection (`sumolib.net.readNet(...).convertXY2LonLat`) rather than hand-typing
approximate coordinates. Import libsumo before pyproj, same DLL-collision-avoidance
order this project has followed since Week 4/5 -- verified no collision between the two
specifically before relying on it here.

Usage:
    python scripts/export_corridor_geometry.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import libsumo  # noqa: F401  -- import before sumolib/pyproj, established DLL-order fix
import sumolib  # noqa: E402

from atlas.common.config import project_root  # noqa: E402

CORRIDOR_JUNCTIONS = ["42430333", "42430338", "42430342", "42430344", "42430347"]
NET_FILE = project_root() / "data" / "networks" / "manhattan_midtown" / "net.net.xml"
OUTPUT_FILE = project_root() / "frontend" / "src" / "corridor_geometry.json"


def export() -> None:
    net = sumolib.net.readNet(str(NET_FILE))
    junctions = []
    for junction_id in CORRIDOR_JUNCTIONS:
        node = net.getNode(junction_id)
        x, y = node.getCoord()
        lon, lat = net.convertXY2LonLat(x, y)
        junctions.append({"intersection_id": junction_id, "lon": lon, "lat": lat})

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps({"junctions": junctions}, indent=2) + "\n")
    print(f"Wrote {len(junctions)} real junction coordinates to {OUTPUT_FILE}")


if __name__ == "__main__":
    export()
