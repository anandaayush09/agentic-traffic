#!/usr/bin/env python
"""Builds a SUMO network (`net.net.xml`) from a city's OSM extract with `netconvert`.

    python scripts/build_network.py --city manhattan [--force]

The city-scale counterpart of `build_district_network.py` (which stays the Midtown entry point):
the same deterministic netconvert flags, applied to `data/networks/<city>/raw.osm.xml`
(see `fetch_city_osm.py`), plus options that only matter at borough scale:

- `--keep-edges.by-vclass passenger`: drop roads cars can't use (pedestrian ways, cycleways, ...);
- `--no-turnarounds`/`--no-internal-links` are NOT used: turn-arounds and internal links keep junction
  behaviour faithful, and the Midtown flags are what the controllers were developed against.

netconvert is single-threaded and, for a whole borough, takes minutes to tens of minutes and a few GB of
RAM; progress goes to `<city>/netconvert.log`. A finished build is a no-op unless `--force`.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atlas.common.config import project_root  # noqa: E402
from atlas.common.sumo_locate import get_binary_path, get_sumo_home  # noqa: E402
from build_district_network import NETCONVERT_FLAGS  # noqa: E402

CITY_EXTRA_FLAGS = ["--keep-edges.by-vclass", "passenger"]


def build_command(raw_osm: Path, net_file: Path, typemap: Path) -> list[str]:
    return [
        str(get_binary_path("netconvert")),
        "--osm-files",
        str(raw_osm),
        "--output-file",
        str(net_file),
        "--type-files",
        str(typemap),
        *NETCONVERT_FLAGS,
        *CITY_EXTRA_FLAGS,
    ]


def build(city: str, force: bool = False) -> Path:
    city_dir = project_root() / "data" / "networks" / city
    raw = city_dir / "raw.osm.xml"
    net = city_dir / "net.net.xml"
    if not raw.is_file():
        raise SystemExit(f"{raw} not found -- run: python scripts/fetch_city_osm.py --city {city}")
    if net.is_file() and not force:
        print(f"{net} already exists ({net.stat().st_size / 1e6:.1f} MB); use --force to rebuild")
        return net

    typemap = get_sumo_home() / "data" / "typemap" / "osmNetconvert.typ.xml"
    cmd = build_command(raw, net, typemap)
    log_path = city_dir / "netconvert.log"
    print(f"Running netconvert on {raw.stat().st_size / 1e6:.1f} MB of OSM ... (log: {log_path})")
    started = time.monotonic()
    tmp = net.with_suffix(".xml.part")
    cmd[cmd.index("--output-file") + 1] = str(tmp)
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0 or not tmp.is_file():
        tmp.unlink(missing_ok=True)
        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-15:])
        raise SystemExit(f"netconvert failed ({result.returncode}); last log lines:\n{tail}")
    tmp.replace(net)
    print(f"Built {net} ({net.stat().st_size / 1e6:.1f} MB) in {time.monotonic() - started:.0f}s")
    return net


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--city", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build(args.city, args.force)


if __name__ == "__main__":
    main()
