#!/usr/bin/env python
"""Audits a built SUMO network: how big is it, and how much of it can ATLAS actually control?

    python scripts/audit_network.py --net data/networks/manhattan/net.net.xml [--json out.json]

This is what sizes city-scale work, so it measures instead of assuming:

- network size: roads, lanes, lane-kilometres, junction types, signalized-junction count;
- per signal, whether ATLAS's controllers can drive it (it has controlled movements and at least one
  controllable green phase -- the same rule the controllers apply) and, if not, why;
- the distributions the control stack's fixed sizes must be derived from: incoming lanes per signal
  (the policy observation is padded to `max_lanes`) and green phases per signal (`max_green_phases`).
  The Midtown district happened to have max 10 lanes and exactly 2 phases; a whole borough will not.

Single streaming pass (`iterparse`, clearing elements as it goes): a city network is tens to hundreds
of MB of XML and this must not hold it in memory (the target machine has 8 GB).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atlas.control.phases import green_phase_indices  # noqa: E402


def audit(net_file: Path) -> dict[str, Any]:
    """Statistics for `net_file` (see module docstring). Pure function of the file."""
    edges = lanes = 0
    lane_length_m = 0.0
    lane_counts: Counter[int] = Counter()
    junction_types: Counter[str] = Counter()
    # tls id -> distinct incoming lanes / phase state strings / logic type
    incoming: dict[str, set[str]] = defaultdict(set)
    phases: dict[str, list[str]] = {}
    logic_type: dict[str, str] = {}

    for _, element in ElementTree.iterparse(net_file, events=("end",)):
        tag = element.tag
        if tag == "edge":
            if element.get("function") != "internal":
                edges += 1
                edge_lanes = element.findall("lane")
                lanes += len(edge_lanes)
                lane_counts[len(edge_lanes)] += 1
                lane_length_m += sum(float(lane.get("length", "0")) for lane in edge_lanes)
            element.clear()
        elif tag == "junction":
            junction_types[element.get("type", "?")] += 1
            element.clear()
        elif tag == "tlLogic":
            tls_id = element.get("id", "")
            phases[tls_id] = [p.get("state", "") for p in element.findall("phase")]
            logic_type[tls_id] = element.get("type", "?")
            element.clear()
        elif tag == "connection":
            tls_id = element.get("tl")
            if tls_id:
                incoming[tls_id].add(f"{element.get('from')}_{element.get('fromLane')}")
            element.clear()

    controllable: list[str] = []
    skipped: dict[str, str] = {}
    lane_hist: Counter[int] = Counter()
    green_hist: Counter[int] = Counter()
    for tls_id, states in phases.items():
        n_in = len(incoming.get(tls_id, ()))
        greens = green_phase_indices(states)
        if n_in == 0:
            skipped[tls_id] = "no controlled connections"
        elif not greens:
            skipped[tls_id] = "no controllable green phase"
        else:
            controllable.append(tls_id)
            lane_hist[n_in] += 1
            green_hist[len(greens)] += 1

    max_lanes = max(lane_hist, default=0)
    max_green = max(green_hist, default=0)
    return {
        "net_file": str(net_file),
        "net_file_mb": round(net_file.stat().st_size / 1e6, 1),
        "roads": edges,
        "lanes": lanes,
        "lane_km": round(lane_length_m / 1000, 3),
        "roads_by_lane_count": dict(sorted(lane_counts.items())),
        "junctions_by_type": dict(junction_types.most_common()),
        "signals_total": len(phases),
        "signals_controllable": len(controllable),
        "signals_skipped": skipped,
        "signal_logic_types": dict(Counter(logic_type.values())),
        "incoming_lanes_per_signal": dict(sorted(lane_hist.items())),
        "green_phases_per_signal": dict(sorted(green_hist.items())),
        "required_max_lanes": max_lanes,
        "required_max_green_phases": max_green,
        "signals_over_district_limits": sum(n for k, n in lane_hist.items() if k > 10)
        + sum(n for k, n in green_hist.items() if k > 2),
        "controllable_ids": sorted(controllable),
    }


def render(report: dict[str, Any]) -> str:
    skipped = report["signals_skipped"]
    lines = [
        f"Network: {report['net_file']} ({report['net_file_mb']} MB)",
        f"  roads {report['roads']:,}   lanes {report['lanes']:,}   lane-km {report['lane_km']:,}",
        f"  junction types: {report['junctions_by_type']}",
        f"  signals: {report['signals_total']:,} total, {report['signals_controllable']:,} "
        f"controllable by ATLAS, {len(skipped):,} skipped",
        f"  signal logic types: {report['signal_logic_types']}",
        f"  incoming lanes per signal (lanes: signals): {report['incoming_lanes_per_signal']}",
        f"  green phases per signal (phases: signals): {report['green_phases_per_signal']}",
        f"  => the control stack must be sized for max_lanes >= {report['required_max_lanes']}, "
        f"max_green_phases >= {report['required_max_green_phases']} "
        "(the Midtown district used 10 and 2)",
    ]
    if skipped:
        reasons = Counter(skipped.values())
        lines.append(f"  skipped signals by reason: {dict(reasons)}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--net", type=Path, required=True)
    parser.add_argument("--json", type=Path, help="also write the full report as JSON")
    args = parser.parse_args()
    report = audit(args.net)
    print(render(report))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nFull report: {args.json}")


if __name__ == "__main__":
    main()
