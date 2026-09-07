"""The Supervisor Agent's LangGraph state machine (docs/design/04_agent_architecture.md
section 4.4): "CLASSIFY -> RETRIEVE_PLAYBOOKS -> PARAMETERIZE -> VERIFY(what-if) ->
{AUTO_APPLY if low-impact | REQUEST_APPROVAL} -> MONITOR -> STAND_DOWN". "Every claim
about consequences must cite a what-if run id -- the graph enforces tool-grounding by
construction (a proposal node cannot be reached without a verification node)": here,
that's structural -- `parameterize` always runs before `verify`, and `decide` (the
AUTO_APPLY/REQUEST_APPROVAL fork) only ever runs after `verify` has populated
`what_if_result`, never before.

AUTO_APPLY vs REQUEST_APPROVAL is implemented as a DYNAMIC decision inside one node
(`decide`), using LangGraph's `interrupt()` -- verified directly against the installed
langgraph package (see this module's own smoke-tested API calls) rather than assumed
from documentation: `interrupt()` pauses graph execution and durably checkpoints state
(docs/design/04_agent_architecture.md section 4.4's "LangGraph durable checkpoints --
survives restart mid-incident"), resumable later with `Command(resume=...)` carrying
the operator's decision. This is a real fork in behavior, not just two node names: the
low-impact path never pauses at all.

Auto-apply policy (deliberately conservative, matching what_if.py's own honesty): a
proposal only auto-applies if its what-if check actually RAN (`checked=True`) and
showed improvement. An unchecked playbook (anything but switch_to_max_pressure/
hold_current_plan, today) ALWAYS goes to REQUEST_APPROVAL -- "no grounded evidence
either way" is never treated as "assume it's fine".
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from atlas.cognitive.playbooks import PlaybookProposal, PlaybookRegistry
from atlas.cognitive.provider_chain import ProviderChain
from atlas.cognitive.what_if import WhatIfResult, run_what_if
from atlas.common.logging import get_logger

logger = get_logger(__name__)


class SupervisorState(TypedDict):
    incident: dict[str, object]
    available_playbooks: list[str]
    proposal: dict[str, object] | None
    provider_used: str | None
    what_if_checked: bool
    what_if_improved: bool | None
    auto_applied: bool
    approved: bool | None
    status: str


def build_supervisor_graph(
    provider_chain: ProviderChain,
    registry: PlaybookRegistry,
    net_file: Path,
    route_file: Path,
    checkpointer: BaseCheckpointSaver[str],
) -> CompiledStateGraph[SupervisorState, None, SupervisorState, SupervisorState]:
    def classify(state: SupervisorState) -> dict[str, object]:
        return {"status": "triaged"}

    def retrieve_playbooks(state: SupervisorState) -> dict[str, object]:
        return {"available_playbooks": registry.names()}

    def parameterize(state: SupervisorState) -> dict[str, object]:
        proposal, provider_used = provider_chain.propose(
            state["incident"], state["available_playbooks"]
        )
        return {"proposal": proposal.model_dump(), "provider_used": provider_used}

    def verify(state: SupervisorState) -> dict[str, object]:
        assert state["proposal"] is not None  # parameterize always runs first
        proposal = PlaybookProposal.model_validate(state["proposal"])
        result: WhatIfResult = run_what_if(proposal, net_file, route_file)
        return {"what_if_checked": result.checked, "what_if_improved": result.improved}

    def decide(state: SupervisorState) -> dict[str, object]:
        low_impact = state["what_if_checked"] and state["what_if_improved"] is True
        if low_impact:
            return {"auto_applied": True, "approved": True, "status": "responding"}

        assert state["proposal"] is not None
        decision = interrupt(
            {
                "incident_id": state["incident"].get("incident_id"),
                "proposal": state["proposal"],
                "what_if_checked": state["what_if_checked"],
                "what_if_improved": state["what_if_improved"],
                "question": "Approve this playbook proposal?",
            }
        )
        approved = bool(decision)
        return {
            "auto_applied": False,
            "approved": approved,
            "status": "responding" if approved else "triaged",
        }

    def monitor(state: SupervisorState) -> dict[str, object]:
        if not state["approved"]:
            return {}
        return {"status": "monitoring"}

    def stand_down(state: SupervisorState) -> dict[str, object]:
        if not state["approved"]:
            return {}
        return {"status": "cleared"}

    graph: StateGraph[SupervisorState, None, SupervisorState, SupervisorState] = StateGraph(
        SupervisorState
    )
    graph.add_node("classify", classify)
    graph.add_node("retrieve_playbooks", retrieve_playbooks)
    graph.add_node("parameterize", parameterize)
    graph.add_node("verify", verify)
    graph.add_node("decide", decide)
    graph.add_node("monitor", monitor)
    graph.add_node("stand_down", stand_down)

    graph.add_edge(START, "classify")
    graph.add_edge("classify", "retrieve_playbooks")
    graph.add_edge("retrieve_playbooks", "parameterize")
    graph.add_edge("parameterize", "verify")
    graph.add_edge("verify", "decide")
    graph.add_edge("decide", "monitor")
    graph.add_edge("monitor", "stand_down")
    graph.add_edge("stand_down", END)

    return graph.compile(checkpointer=checkpointer)


def initial_state(incident: dict[str, object]) -> SupervisorState:
    return SupervisorState(
        incident=incident,
        available_playbooks=[],
        proposal=None,
        provider_used=None,
        what_if_checked=False,
        what_if_improved=None,
        auto_applied=False,
        approved=None,
        status="detected",
    )
