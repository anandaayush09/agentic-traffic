#!/usr/bin/env python
"""Fetch and build the Week 2 real district network: Midtown Manhattan, W 42nd-48th St
between 6th and 9th Ave -- chosen purely for OpenStreetMap data quality (complete signal
tags, clean orthogonal grid, no diagonal streets complicating junction geometry), per the
selection criteria in docs/design/08_data_strategy.md section 8.1. This replaces the
Week 1 sandbox grid as the network used for demand generation, scenarios, and training.

Two independent steps, matched to how trustworthy/deterministic each one is:

- `fetch_osm_extract`: downloads the raw OSM extract from the Overpass API. This hits a
  network dependency, and OSM data changes over time -- re-running it later would
  silently drift the district's geometry. So the fetched extract is committed to git
  (data/networks/manhattan_midtown/raw.osm.xml, ~115KB) as a dated snapshot, and this
  function is a no-op unless `--refetch` is passed explicitly. Nothing in the test suite
  or CI ever triggers a live fetch.
- `build_network`: runs SUMO's `netconvert` on the committed raw extract to produce
  net.net.xml. This is fully deterministic given the flags below, so it is *not*
  committed -- it's a build artifact, regenerated the same way scripts/
  generate_sandbox_network.py regenerates the Week 1 placeholder network.

Usage:
    python scripts/build_district_network.py            # build from the committed extract
    python scripts/build_district_network.py --refetch   # re-download from Overpass first
    python scripts/build_district_network.py --force     # rebuild net.net.xml even if present
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atlas.common.config import project_root  # noqa: E402
from atlas.common.logging import configure_logging, get_logger  # noqa: E402
from atlas.common.sumo_locate import get_binary_path, get_sumo_home  # noqa: E402

configure_logging()
logger = get_logger(__name__)

OUTPUT_DIR = project_root() / "data" / "networks" / "manhattan_midtown"
RAW_OSM_FILE = OUTPUT_DIR / "raw.osm.xml"
NET_FILE = OUTPUT_DIR / "net.net.xml"

# Bounding box (south, west, north, east), WGS84. Trimmed from an initial 8-street x
# 4-avenue extract (40 traffic lights, more than the ~25-intersection design target) down
# to 6 streets x 3 avenues to land close to spec -- see AUDIT.md for the full trade-off.
BBOX = (40.7545, -73.9895, 40.7595, -73.9810)
OVERPASS_URL = "https://overpass.kumi.systems/api/interpreter"
HIGHWAY_TYPES = "primary|secondary|tertiary|primary_link|secondary_link|tertiary_link|residential"

# netconvert flags, fixed here (not CLI options) for the same reason the sandbox
# network's grid parameters are constants: this is a versioned fixture, not a tunable.
NETCONVERT_FLAGS = [
    "--geometry.remove",
    "--roundabouts.guess",
    "--ramps.guess",
    "--junctions.join",
    "--tls.guess-signals",
    "--tls.discard-simple",
    "--tls.join",
    "--output.original-names",
    "--junctions.corner-detail",
    "5",
    "--remove-edges.isolated",
]


def fetch_osm_extract(force: bool = False) -> None:
    if RAW_OSM_FILE.is_file() and not force:
        logger.info("district_network.osm_extract_present", path=str(RAW_OSM_FILE))
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    south, west, north, east = BBOX
    query = (
        "[out:xml][timeout:90];"
        f'(way["highway"~"^({HIGHWAY_TYPES})$"]({south},{west},{north},{east}););'
        "(._;>;);out meta;"
    )
    logger.info("district_network.fetching_osm", url=OVERPASS_URL, bbox=BBOX)
    data = urllib.parse.urlencode({"data": query}).encode("utf-8")
    request = urllib.request.Request(OVERPASS_URL, data=data, method="POST")
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
        body = response.read()

    if b"<osm " not in body[:200]:
        raise RuntimeError(
            "Overpass response does not look like OSM XML (likely an error page); "
            "not overwriting any existing extract."
        )

    RAW_OSM_FILE.write_bytes(body)
    logger.info("district_network.osm_extract_saved", path=str(RAW_OSM_FILE), bytes=len(body))


def build_network(force: bool = False) -> None:
    if not RAW_OSM_FILE.is_file():
        raise RuntimeError(
            f"{RAW_OSM_FILE} not found. Run with --refetch, or check out the committed "
            "extract from git."
        )
    if NET_FILE.is_file() and not force:
        logger.info("district_network.net_file_present", path=str(NET_FILE))
        return

    sumo_home = get_sumo_home()
    typemap = sumo_home / "data" / "typemap" / "osmNetconvert.typ.xml"
    cmd = [
        str(get_binary_path("netconvert")),
        "--osm-files",
        str(RAW_OSM_FILE),
        "--output-file",
        str(NET_FILE),
        "--type-files",
        str(typemap),
        *NETCONVERT_FLAGS,
    ]
    logger.info("district_network.building_net", cmd=" ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        logger.error(
            "district_network.netconvert_failed", stdout=result.stdout, stderr=result.stderr
        )
        raise RuntimeError(f"netconvert failed ({result.returncode})")

    if not NET_FILE.is_file():
        raise RuntimeError("netconvert reported success but net.net.xml is missing.")
    logger.info("district_network.done", path=str(NET_FILE))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refetch", action="store_true", help="Re-download the OSM extract from Overpass."
    )
    parser.add_argument(
        "--force", action="store_true", help="Rebuild net.net.xml even if it already exists."
    )
    args = parser.parse_args()

    fetch_osm_extract(force=args.refetch)
    build_network(force=args.force or args.refetch)


if __name__ == "__main__":
    main()
