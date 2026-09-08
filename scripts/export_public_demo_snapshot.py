#!/usr/bin/env python
"""Exports a REAL, frozen snapshot of the scripted incident suite (Week 9's
SCRIPTED_INCIDENTS, run through the real Supervisor graph against the real
manhattan_midtown corridor) for the Week 10 public read-only demo
(docs/design/10_deployment.md: "Read-only replay: frontend + dashboard_api +
Postgres snapshot... zero control endpoints, no bus, no LLM keys").

Scope, deliberately: the frozen design's public demo needs a Postgres-backed
telemetry/replay store that doesn't exist yet (docs/security/threat_model.md section
3.5's own named gap) -- a live `dashboard_api` reading from it is Week 11+ scope, not
guessable today. This script is the honest, buildable-now substitute: a real, static
JSON snapshot of real Supervisor graph runs (not fabricated example data), which the
frontend's static demo mode (frontend/src/api.ts's `loadStaticSnapshot`) reads
directly with zero network calls -- literally "no bus, no LLM keys" since there is no
backend at all to call, and "zero control endpoints" since the static build has no
Approve/Reject wiring (see App.tsx's `STATIC_DEMO` branch).

Usage:
    python scripts/export_public_demo_snapshot.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from atlas.cognitive.incident_suite import SCRIPTED_INCIDENTS  # noqa: E402
from atlas.cognitive.playbooks import PlaybookRegistry  # noqa: E402
from atlas.cognitive.provider_chain import ProviderChain  # noqa: E402
from atlas.cognitive.supervisor import build_supervisor_graph  # noqa: E402
from atlas.common.config import project_root  # noqa: E402
from atlas.services.cognitive_api import CognitiveApiService  # noqa: E402

OUTPUT_FILE = project_root() / "frontend" / "src" / "public_demo_snapshot.json"


def export() -> None:
    network_dir = project_root() / "data" / "networks" / "manhattan_midtown"
    net_file = network_dir / "net.net.xml"
    route_file = network_dir / "demand" / "pm_peak.rou.xml"
    vtypes_file = network_dir / "vtypes.add.xml"

    registry = PlaybookRegistry()
    chain = ProviderChain([])  # no LLM providers -- deterministic mapper fallback
    graph = build_supervisor_graph(
        chain,
        registry,
        net_file,
        route_file,
        InMemorySaver(),
        additional_files=[vtypes_file] if vtypes_file.is_file() else [],
    )
    service = CognitiveApiService(graph)

    for scripted in SCRIPTED_INCIDENTS:
        result = service.submit_incident(scripted.incident)
        print(f"{scripted.incident_id}: status={result['state']['status']}")

    incidents = service.list_incidents()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(incidents, indent=2) + "\n")
    print(f"\nWrote {len(incidents)} real incident results to {OUTPUT_FILE}")


if __name__ == "__main__":
    export()
