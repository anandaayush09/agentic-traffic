#!/usr/bin/env python
"""Runs a real Cognitive API instance for the Week 10 dashboard to talk to during
local development/demoing -- the actual FastAPI app (src/atlas/services/cognitive_api.py)
wired to a real compiled Supervisor graph against the real manhattan_midtown corridor,
not a mock server. Prints a ready-made operator JWT so the dashboard's "Operator token"
field has something real to paste in immediately.

Uses real LLM providers for playbook proposals when GROQ_API_KEY and/or GEMINI_API_KEY
are set (see atlas.cognitive.llm); otherwise proposals come from the deterministic mapper.

Also starts the live digital twin (atlas.services.live_traffic) so the dashboard map
shows real simulated traffic, and so submitted accident/closure/construction/weather
incidents visibly disrupt the roads they name. Set ATLAS_LIVE_TRAFFIC=0 to run without
it (incident approval only, no map traffic).

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

from atlas.cognitive.llm import build_llm_chain  # noqa: E402
from atlas.cognitive.playbooks import PlaybookRegistry  # noqa: E402
from atlas.cognitive.supervisor import build_supervisor_graph  # noqa: E402
from atlas.common.config import project_root  # noqa: E402
from atlas.services.auth import create_operator_token  # noqa: E402
from atlas.services.cognitive_api import (  # noqa: E402
    CognitiveApiService,
    LiveTraffic,
    create_app,
)

DEMO_JWT_SECRET = "local-demo-secret-do-not-use-in-prod-32-bytes"


def _start_live_traffic(net_file: Path, network_dir: Path, vtypes_file: Path) -> LiveTraffic | None:
    """Starts the live twin, or returns None (with the reason printed) rather than
    failing -- the live map is an enhancement, the approval flow must still come up."""
    if os.environ.get("ATLAS_LIVE_TRAFFIC", "1") == "0":
        print("Live traffic: disabled (ATLAS_LIVE_TRAFFIC=0)")
        return None
    try:
        from atlas.services.live_traffic import LiveTrafficService

        service = LiveTrafficService(
            net_file,
            network_dir / "demand" / "pm_peak.rou.xml",
            [vtypes_file] if vtypes_file.is_file() else [],
        )
    except Exception as exc:  # e.g. pyproj (a dev-extra dependency) not installed
        print(f"Live traffic: disabled ({type(exc).__name__}: {exc})")
        return None
    service.start()
    print("Live traffic: simulation starting in the background (map fills in ~5s)")
    return service


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
    # Groq / Gemini if their keys are set (GROQ_API_KEY / GEMINI_API_KEY, e.g. in .env),
    # ranked per incident by live quota + health with Redis-shared limits; otherwise just
    # the deterministic mapper. Never prints a key.
    llm = build_llm_chain(registry=registry)
    for note in llm.notes:
        print(note)
    chain = llm.chain
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

    live_traffic = _start_live_traffic(net_file, network_dir, vtypes_file)

    service = CognitiveApiService(graph)
    app = create_app(
        service, jwt_secret, dashboard_origins=dashboard_origins, live_traffic=live_traffic
    )

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
