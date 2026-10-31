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
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import uvicorn  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from atlas.cognitive.llm import build_llm_chain  # noqa: E402
from atlas.cognitive.persistence import open_durable_state  # noqa: E402
from atlas.cognitive.playbooks import PlaybookRegistry  # noqa: E402
from atlas.cognitive.supervisor import build_supervisor_graph  # noqa: E402
from atlas.common.config import ProjectSettings, project_root  # noqa: E402
from atlas.common.logging import configure_logging  # noqa: E402
from atlas.common.observability import init_sentry  # noqa: E402
from atlas.common.runtime_config import (  # noqa: E402
    DEMO_JWT_SECRET,
    app_version,
    environment_name,
    is_protected,
    validate_runtime_config,
)
from atlas.services.auth import create_operator_token  # noqa: E402
from atlas.services.cognitive_api import (  # noqa: E402
    CognitiveApiService,
    LiveTraffic,
    create_app,
)
from atlas.twin.tiles import TileStore  # noqa: E402


def _start_live_traffic(
    net_file: Path, network_dir: Path, vtypes_file: Path, *, city: bool
) -> LiveTraffic | None:
    """Starts the live twin, or returns None (with the reason printed) rather than
    failing -- the live map is an enhancement, the approval flow must still come up."""
    if os.environ.get("ATLAS_LIVE_TRAFFIC", "1") == "0":
        print("Live traffic: disabled (ATLAS_LIVE_TRAFFIC=0)")
        return None
    try:
        from atlas.marl import load_policy
        from atlas.services.live_traffic import LiveTrafficService

        # Optional trained MAPPO policy: adds an "RL" option to every junction's mode menu.
        policy = None
        rl_path = os.environ.get("ATLAS_RL_CHECKPOINT")
        if rl_path:
            policy = load_policy(Path(rl_path))
            print(f"RL mode: ENABLED from {policy.description}")
            print(
                "  (loading says nothing about quality -- evaluate it: "
                "python scripts/run_evaluation_pilot.py --rl-checkpoint <file>)"
            )
        city_options: dict[str, Any] = {}
        if city:
            # A whole city: in-process libsumo (the traci socket cannot keep up with thousands
            # of signals), every controllable signal under ATLAS, and teleporting on so a
            # gridlock clears instead of freezing the map forever.
            city_options = {
                "backend": "libsumo",
                "tls_ids": "all",
                "time_to_teleport_s": 300.0,
                "scale": float(os.environ.get("ATLAS_DEMAND_SCALE", "0.25")),
            }
        service = LiveTrafficService(
            net_file,
            network_dir / "demand" / "pm_peak.rou.xml",
            [vtypes_file] if vtypes_file.is_file() else [],
            policy=policy,
            **city_options,
        )
    except Exception as exc:  # e.g. pyproj (a dev-extra dependency) not installed
        print(f"Live traffic: disabled ({type(exc).__name__}: {exc})")
        return None
    service.start()
    print(
        "Live traffic: simulation starting in the background (map fills in ~5s); "
        f"ATLAS controls {len(service.junction_positions())} signalized junctions"
    )
    return service


def main() -> None:
    # Fail fast before doing any work: a protected environment (staging/production) must not
    # boot with the public demo secret, the demo-token endpoint, or default CORS origins.
    problems = validate_runtime_config(os.environ)
    if problems:
        print(f"Refusing to start in ATLAS_ENV={environment_name(os.environ)}:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        raise SystemExit(2)

    # ATLAS_LOG_FORMAT=json in containers (one JSON object per line, searchable); console locally.
    settings = ProjectSettings()
    configure_logging(settings.log_level, settings.log_format)
    version, environment = app_version(os.environ), environment_name(os.environ)
    if init_sentry(os.environ, release=version, environment=environment):
        print("Error monitoring: Sentry enabled")
    print(f"ATLAS {version} starting in '{environment}'")

    # The real manhattan_midtown corridor, not the sandbox network -- frontend/src/
    # MapView.tsx plots the real corridor junctions from
    # scripts/export_corridor_geometry.py, so a demo incident's junction_ids must
    # resolve against the SAME network those coordinates came from, not a
    # differently-shaped sandbox network (an earlier version of this script used the
    # sandbox network here while the frontend's demo incident referenced a real
    # manhattan junction id -- a real "no connections found for traffic light" error
    # caught by actually clicking through the dashboard, not assumed to work).
    network_name = os.environ.get("ATLAS_NETWORK", "manhattan_midtown")
    city = network_name != "manhattan_midtown"
    network_dir = project_root() / "data" / "networks" / network_name
    if not (network_dir / "net.net.xml").is_file():
        raise SystemExit(f"No built network at {network_dir} (ATLAS_NETWORK={network_name})")
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
    # Opt-in durable decisions: with ATLAS_STATE_DB=<file> incidents (including ones paused
    # for approval) survive an API restart; without it they live in memory as before.
    state_db = os.environ.get("ATLAS_STATE_DB")
    if state_db:
        durable = open_durable_state(Path(state_db))
        checkpointer, incident_index = durable.checkpointer, durable.index
        print(f"Incident history: durable at {durable.path} ({len(incident_index.all())} stored)")
    else:
        checkpointer, incident_index = InMemorySaver(), None
        print("Incident history: in memory (set ATLAS_STATE_DB=<file> to keep it across restarts)")
    graph = build_supervisor_graph(
        chain,
        registry,
        net_file,
        route_file,
        checkpointer,
        additional_files=[vtypes_file] if vtypes_file.is_file() else [],
        # City: what-if runs on a small local tile around the incident, in a separate process (a
        # whole-borough what-if is two full simulations per request, and the live twin already owns
        # the one in-process libsumo). The first incident in an area builds its tile (~15 s); tiles
        # are cached under <network>/tiles.
        tiles=TileStore(net_file, route_file, network_dir / "tiles") if city else None,
    )
    # Overridable so the same script serves both local dev (the hardcoded demo
    # secret, printed below, is fine to know) and a kind/Helm deployment (deploy/
    # helm/atlas/values-kind.yaml sets a real secret via env var) without a second
    # entrypoint to keep in sync.
    jwt_secret = os.environ.get("ATLAS_JWT_SECRET", DEMO_JWT_SECRET)
    dashboard_origins_env = os.environ.get("ATLAS_DASHBOARD_ORIGINS")
    dashboard_origins = dashboard_origins_env.split(",") if dashboard_origins_env else None

    live_traffic = _start_live_traffic(net_file, network_dir, vtypes_file, city=city)

    # The dashboard's "Get demo token" button -- local dev only (see create_app).
    demo_auth = os.environ.get("ATLAS_DEMO_AUTH") == "1"
    if demo_auth:
        print(
            "Demo token endpoint: ENABLED (local clients only) -- dev convenience, not for sharing"
        )

    service = CognitiveApiService(graph, index=incident_index)
    app = create_app(
        service,
        jwt_secret,
        dashboard_origins=dashboard_origins,
        live_traffic=live_traffic,
        demo_token_enabled=demo_auth,
        version=version,
        environment=environment,
    )

    print("=== Cognitive API demo server ===")
    if is_protected(os.environ):
        # Never print a token here: stdout goes to the container's logs, and a valid operator
        # token in logs hands operator rights to anyone who can read them. Mint one on purpose:
        #   ATLAS_JWT_SECRET=... python scripts/mint_operator_token.py --subject you --hours 1
        print(
            "Operator tokens are NOT printed in this environment (see scripts/mint_operator_token.py)"
        )
    else:
        token = create_operator_token("demo-operator", jwt_secret, expires_in_s=8 * 3600.0)
        print("Operator JWT (paste into the dashboard's 'Operator token' field):")
        print(token)
    print()

    # 0.0.0.0, not 127.0.0.1: this same script runs both on a developer's machine
    # (reachable via localhost either way) and inside a container/pod (must bind all
    # interfaces to be reachable from the Service in front of it). $PORT is what platforms
    # such as Render/Cloud Run inject; ATLAS_API_PORT overrides for everything else.
    port = int(os.environ.get("PORT") or os.environ.get("ATLAS_API_PORT") or 8000)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")  # noqa: S104


if __name__ == "__main__":
    main()
