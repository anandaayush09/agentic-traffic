#!/usr/bin/env python
"""Downloads a whole city's drivable road network from OpenStreetMap (Overpass API) for
city-scale ATLAS work, and records exactly what was fetched so the build is reproducible.

    python scripts/fetch_city_osm.py --city manhattan [--force]

Writes (gitignored, large):  data/networks/<city>/raw.osm.xml
Writes (committed, tiny):    data/networks/<city>/SOURCE.json  -- the query, the date, byte size and
sha256 of the extract. OSM changes continuously, so re-fetching later gives different geometry; the
record lets you tell which snapshot a given result came from, the same reason the Midtown extract
is committed (see build_district_network.py).

It is a no-op if the extract already exists (pass --force to re-download), streams to disk (the
file is tens of MB; nothing is held in memory) and writes to a temporary file first, so an
interrupted download can never leave a truncated file that looks complete.

Be a good Overpass citizen: one request, a descriptive User-Agent, and no automatic retry loop
hammering a public server.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atlas.common.config import project_root  # noqa: E402

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
USER_AGENT = "atlas-traffic-research/0.1 (open-source traffic-control research project)"

# Roads a car can use and a signal controller cares about. `service` roads (parking aisles,
# driveways) are left out deliberately: they are most of the data, carry no signals, and would
# multiply the network size for nothing.
HIGHWAY_TYPES = (
    "motorway|trunk|primary|secondary|tertiary|unclassified|residential|"
    "motorway_link|trunk_link|primary_link|secondary_link|tertiary_link|living_street"
)

# OSM relation ids of administrative areas (verified against Overpass: the relation's `name` tag).
CITIES: dict[str, dict[str, object]] = {
    "manhattan": {"relation": 8398124, "label": "Manhattan (New York County borough)"},
}


def build_query(relation: int, timeout_s: int = 900) -> str:
    return (
        f"[out:xml][timeout:{timeout_s}];"
        f"rel({relation});map_to_area->.a;"
        f'(way["highway"~"^({HIGHWAY_TYPES})$"](area.a););'
        "(._;>;);out;"
    )


def fetch(city: str, force: bool = False) -> Path:
    spec = CITIES[city]
    out_dir = project_root() / "data" / "networks" / city
    raw = out_dir / "raw.osm.xml"
    if raw.is_file() and not force:
        print(
            f"{raw} already exists ({raw.stat().st_size / 1e6:.1f} MB); use --force to re-download"
        )
        return raw

    out_dir.mkdir(parents=True, exist_ok=True)
    query = build_query(int(spec["relation"]))  # type: ignore[call-overload]
    request = urllib.request.Request(  # noqa: S310 -- fixed https constant
        OVERPASS_URL,
        data=("data=" + urllib.request.quote(query)).encode(),
        headers={"User-Agent": USER_AGENT, "Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    tmp = raw.with_suffix(".xml.part")
    sha, size, started = hashlib.sha256(), 0, time.monotonic()
    print(f"Fetching {spec['label']} from {OVERPASS_URL} ...", flush=True)
    with urllib.request.urlopen(request, timeout=1200) as response, tmp.open("wb") as handle:  # noqa: S310
        while chunk := response.read(1 << 20):
            handle.write(chunk)
            sha.update(chunk)
            size += len(chunk)
            print(f"  {size / 1e6:7.1f} MB  ({time.monotonic() - started:5.0f}s)", flush=True)

    head = tmp.read_bytes()[:400]
    if b"<osm " not in head or b"<remark>" in tmp.read_bytes()[-400:]:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            "Overpass did not return a complete OSM document (error page or a runtime-limit "
            "remark); nothing was saved."
        )
    os.replace(tmp, raw)

    (out_dir / "SOURCE.json").write_text(
        json.dumps(
            {
                "city": city,
                "label": spec["label"],
                "overpass_url": OVERPASS_URL,
                "query": query,
                "fetched_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                "bytes": size,
                "sha256": sha.hexdigest(),
                "license": "Data (c) OpenStreetMap contributors, ODbL 1.0",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved {raw} ({size / 1e6:.1f} MB, sha256 {sha.hexdigest()[:16]}...)")
    return raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--city", choices=sorted(CITIES), required=True)
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()
    fetch(args.city, args.force)


if __name__ == "__main__":
    main()
