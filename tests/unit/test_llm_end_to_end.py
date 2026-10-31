"""End to end through the real pieces, with only the network mocked: an LLM proposal (from a
mocked Groq endpoint) flows through the quota-aware chain into the real Supervisor graph,
which runs the REAL SUMO what-if on the Manhattan network and pauses for human approval.
Also proves a hostile reply cannot slip past validation and that a failing provider falls
through to the next one -- all the way to a real simulation.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from langgraph.checkpoint.memory import InMemorySaver

from atlas.cognitive.llm import LlmSettings, build_llm_chain
from atlas.cognitive.llm.limits import InMemoryRateLimitStore
from atlas.cognitive.playbooks import PlaybookRegistry
from atlas.cognitive.supervisor import build_supervisor_graph, initial_state
from atlas.common.config import project_root

NETWORK_DIR = project_root() / "data" / "networks" / "manhattan_midtown"
NET_FILE = NETWORK_DIR / "net.net.xml"
ROUTE_FILE = NETWORK_DIR / "demand" / "pm_peak.rou.xml"
VTYPES_FILE = NETWORK_DIR / "vtypes.add.xml"

pytestmark = pytest.mark.skipif(
    not (NET_FILE.is_file() and ROUTE_FILE.is_file()), reason="manhattan_midtown network not built"
)

JUNCTION = "42430342"
INCIDENT: dict[str, object] = {
    "incident_id": "llm-e2e-1",
    "class": "accident",
    "status": "detected",
    "confidence": 0.9,
    "location": {"link_ids": [JUNCTION]},
}


def _graph(client: httpx.Client, **keys: str):  # type: ignore[no-untyped-def]
    registry = PlaybookRegistry()
    setup = build_llm_chain(
        LlmSettings(_env_file=None, **keys),  # type: ignore[arg-type]
        registry=registry,
        store=InMemoryRateLimitStore(),
        client=client,
    )
    graph = build_supervisor_graph(
        setup.chain, registry, NET_FILE, ROUTE_FILE, InMemorySaver(), additional_files=[VTYPES_FILE]
    )
    return graph


def _groq_ok(playbook: dict[str, object]) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(playbook)}}]})


class TestLlmProposalThroughTheRealSupervisor:
    def test_groq_proposal_is_verified_by_a_real_simulation_then_awaits_approval(self) -> None:
        reply = {
            "playbook_name": "divert_arterial",
            "params": {"from_edge": JUNCTION, "to_edge": JUNCTION, "duration_s": 600},
            "rationale": "Divert traffic away from the crash site.",
        }
        client = httpx.Client(transport=httpx.MockTransport(lambda _: _groq_ok(reply)))
        graph = _graph(client, groq_api_key="gsk_test")

        state = graph.invoke(
            initial_state(dict(INCIDENT)), config={"configurable": {"thread_id": "e2e-1"}}
        )

        assert state["provider_used"] == "groq"
        assert state["proposal"]["rationale"] == "Divert traffic away from the crash site."
        assert state["what_if_checked"] is True  # a real SUMO comparison ran on the LLM's plan
        assert state["what_if_detail"]["metric"] == "mean_halted_vehicles"
        assert state["auto_applied"] is False  # rerouting still asks a human
        assert "__interrupt__" in state

    def test_a_proposal_naming_an_invented_road_never_reaches_the_simulation(self) -> None:
        hostile = {
            "playbook_name": "divert_arterial",
            "params": {"from_edge": "not-a-real-edge", "to_edge": JUNCTION, "duration_s": 600},
            "rationale": "Trust me.",
        }
        client = httpx.Client(transport=httpx.MockTransport(lambda _: _groq_ok(hostile)))
        graph = _graph(client, groq_api_key="gsk_test")

        state = graph.invoke(
            initial_state(dict(INCIDENT)), config={"configurable": {"thread_id": "e2e-2"}}
        )

        # Rejected as ungrounded, so the deterministic mapper answered instead.
        assert state["provider_used"] == "deterministic"
        assert state["proposal"]["params"]["from_edge"] == JUNCTION

    def test_groq_down_falls_through_to_gemini_and_still_gets_a_real_what_if(self) -> None:
        reply = {
            "playbook_name": "divert_arterial",
            "params": {"from_edge": JUNCTION, "to_edge": JUNCTION, "duration_s": 600},
            "rationale": "Gemini's plan.",
        }

        def handler(request: httpx.Request) -> httpx.Response:
            if "groq" in request.url.host:
                return httpx.Response(429, headers={"retry-after": "30"})
            return httpx.Response(
                200, json={"candidates": [{"content": {"parts": [{"text": json.dumps(reply)}]}}]}
            )

        graph = _graph(
            httpx.Client(transport=httpx.MockTransport(handler)),
            groq_api_key="gsk_test",
            gemini_api_key="AIza_test",
        )

        state = graph.invoke(
            initial_state(dict(INCIDENT)), config={"configurable": {"thread_id": "e2e-3"}}
        )

        assert state["provider_used"] == "gemini"
        assert state["what_if_checked"] is True


def test_path_constants_resolve() -> None:
    assert Path(NET_FILE).is_file()
