#!/usr/bin/env python
"""Runs a real Cognitive API instance for the Week 10 dashboard to talk to during
local development/demoing -- the actual FastAPI app (src/atlas/services/cognitive_api.py)
wired to a real compiled Supervisor graph against the real manhattan_midtown corridor,
not a mock server. Prints a ready-made operator JWT so the dashboard's "Operator token"
field has something real to paste in immediately.

Requires data/networks/manhattan_midtown/net.net.xml to already be built (run
scripts/build_district_network.py first if it doesn't exist yet).

Usage:
    python scripts/run_dashboard_demo_api.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import uvicorn  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from atlas.cognitive.playbooks import PlaybookRegistry  # noqa: E402
from atlas.cognitive.provider_chain import ProviderChain  # noqa: E402
from atlas.cognitive.supervisor import build_supervisor_graph  # noqa: E402
from atlas.common.config import project_root  # noqa: E402
from atlas.services.auth import create_operator_token  # noqa: E402
from atlas.services.cognitive_api import CognitiveApiService, create_app  # noqa: E402

DEMO_JWT_SECRET = "local-demo-secret-do-not-use-in-prod-32-bytes"


def main() -> None:
    # The real manhattan_midtown corridor, not the sandbox network -- frontend/src/
    # MapView.tsx plots the real corridor junctions from
    # scripts/export_corridor_geometry.py, so a demo incident's junction_ids must
    # resolve against the SAME network those coordinates came from, not a
    # differently-shaped sandbox network (an earlier version of this script used the
    # sandbox network here while the frontend's demo incident referenced a real
    # manhattan junction id -- a real "no connections found for traffic light" error
    # caught by actually clicking through the dashboard, not assumed to work).
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
    # Overridable so the same script serves both local dev (the hardcoded demo
    # secret, printed below, is fine to know) and a kind/Helm deployment (deploy/
    # helm/atlas/values-kind.yaml sets a real secret via env var) without a second
    # entrypoint to keep in sync.
    jwt_secret = os.environ.get("ATLAS_JWT_SECRET", DEMO_JWT_SECRET)
    dashboard_origins_env = os.environ.get("ATLAS_DASHBOARD_ORIGINS")
    dashboard_origins = dashboard_origins_env.split(",") if dashboard_origins_env else None

    service = CognitiveApiService(graph)
    app = create_app(service, jwt_secret, dashboard_origins=dashboard_origins)

    token = create_operator_token("demo-operator", jwt_secret, expires_in_s=8 * 3600.0)
    print("=== Cognitive API demo server ===")
    print("Operator JWT (paste into the dashboard's 'Operator token' field):")
    print(token)
    print()

    # 0.0.0.0, not 127.0.0.1: this same script runs both on a developer's machine
    # (reachable via localhost either way) and inside a kind pod (must bind all
    # interfaces to be reachable from the Service in front of it).
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")  # noqa: S104


if __name__ == "__main__":
    main()
