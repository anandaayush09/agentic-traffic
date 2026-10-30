"""Validates the Supervisor graph's structure and both AUTO_APPLY/REQUEST_APPROVAL
paths against real LangGraph execution (not a mock of the graph library) -- verified
API calls (StateGraph, interrupt/Command resume, InMemorySaver) against a real running
graph, using a FAKE what-if function so these tests don't need SUMO at all.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from atlas.cognitive.playbooks import PlaybookRegistry
from atlas.cognitive.provider_chain import ProviderChain
from atlas.cognitive.supervisor import build_supervisor_graph, initial_state
from atlas.cognitive.what_if import WhatIfResult

DUMMY_PATH = Path("unused_for_these_tests.net.xml")


def _incident(incident_class: str = "sensor_fault") -> dict[str, object]:
    return {
        "incident_id": "inc-1",
        "class": incident_class,
        "status": "detected",
        "confidence": 0.9,
        "location": {"link_ids": ["B1"]},
    }


def _graph_with_fake_what_if(result: WhatIfResult):  # type: ignore[no-untyped-def]
    """Builds a real graph, but patches run_what_if so no SUMO twin is ever started --
    these tests are about graph routing/state, not the what-if tool itself (see
    tests/unit/test_what_if.py for that)."""
    registry = PlaybookRegistry()
    chain = ProviderChain([])  # empty -> always falls through to the deterministic mapper
    graph = build_supervisor_graph(chain, registry, DUMMY_PATH, DUMMY_PATH, InMemorySaver())
    return patch("atlas.cognitive.supervisor.run_what_if", return_value=result), graph


class TestAutoApplyPath:
    def test_checked_and_improved_auto_applies_without_pausing(self) -> None:
        patcher, graph = _graph_with_fake_what_if(
            WhatIfResult(
                checked=True, improved=True, baseline_mean_pressure=10.0, proposed_mean_pressure=5.0
            )
        )
        with patcher:
            config = {"configurable": {"thread_id": "t1"}}
            result = graph.invoke(initial_state(_incident()), config=config)

        assert "__interrupt__" not in result
        assert result["auto_applied"] is True
        assert result["approved"] is True
        assert result["status"] == "cleared"

    def test_proposal_and_provider_used_are_populated(self) -> None:
        patcher, graph = _graph_with_fake_what_if(
            WhatIfResult(
                checked=True, improved=True, baseline_mean_pressure=1.0, proposed_mean_pressure=0.0
            )
        )
        with patcher:
            result = graph.invoke(
                initial_state(_incident()), config={"configurable": {"thread_id": "t2"}}
            )

        assert result["proposal"]["playbook_name"] == "switch_to_max_pressure"
        assert result["provider_used"] == "deterministic"


class TestRequestApprovalPath:
    def test_unchecked_what_if_pauses_for_approval(self) -> None:
        patcher, graph = _graph_with_fake_what_if(
            WhatIfResult(
                checked=False,
                improved=None,
                baseline_mean_pressure=None,
                proposed_mean_pressure=None,
            )
        )
        with patcher:
            config = {"configurable": {"thread_id": "t3"}}
            paused = graph.invoke(initial_state(_incident()), config=config)

            assert "__interrupt__" in paused
            assert paused["status"] == "triaged"  # not yet applied

            interrupt_payload = paused["__interrupt__"][0].value
            assert interrupt_payload["question"] == "Approve this playbook proposal?"
            assert interrupt_payload["what_if_checked"] is False

            resumed = graph.invoke(Command(resume=True), config=config)

        assert resumed["approved"] is True
        assert resumed["status"] == "cleared"

    def test_checked_but_not_improved_also_requires_approval(self) -> None:
        """Verified worse than baseline -- must NOT auto-apply just because it was
        checked; only checked AND improved auto-applies."""
        patcher, graph = _graph_with_fake_what_if(
            WhatIfResult(
                checked=True,
                improved=False,
                baseline_mean_pressure=5.0,
                proposed_mean_pressure=10.0,
            )
        )
        with patcher:
            paused = graph.invoke(
                initial_state(_incident()), config={"configurable": {"thread_id": "t4"}}
            )

        assert "__interrupt__" in paused

    def test_operator_rejection_leaves_incident_untriaged_not_cleared(self) -> None:
        patcher, graph = _graph_with_fake_what_if(
            WhatIfResult(
                checked=False,
                improved=None,
                baseline_mean_pressure=None,
                proposed_mean_pressure=None,
            )
        )
        with patcher:
            config = {"configurable": {"thread_id": "t5"}}
            graph.invoke(initial_state(_incident()), config=config)

            resumed = graph.invoke(Command(resume=False), config=config)

        assert resumed["approved"] is False
        assert resumed["status"] == "triaged"

    def test_pause_survives_as_a_durable_checkpoint(self) -> None:
        """The literal design claim: state is checkpointed, not held only in memory
        across the pause -- verified via get_state on the compiled graph."""
        patcher, graph = _graph_with_fake_what_if(
            WhatIfResult(
                checked=False,
                improved=None,
                baseline_mean_pressure=None,
                proposed_mean_pressure=None,
            )
        )
        with patcher:
            config = {"configurable": {"thread_id": "t6"}}
            graph.invoke(initial_state(_incident()), config=config)

            snapshot = graph.get_state(config)

        assert snapshot.next == ("decide",)


class TestProposalCarriesRealPlaybookDefaults:
    def test_accident_class_proposes_divert_arterial(self) -> None:
        patcher, graph = _graph_with_fake_what_if(
            WhatIfResult(
                checked=False,
                improved=None,
                baseline_mean_pressure=None,
                proposed_mean_pressure=None,
            )
        )
        with patcher:
            paused = graph.invoke(
                initial_state(_incident("accident")),
                config={"configurable": {"thread_id": "t7"}},
            )

        assert paused["proposal"]["playbook_name"] == "divert_arterial"

    def test_emergency_run_never_pauses_since_hold_current_plan_is_always_safe(self) -> None:
        """hold_current_plan's what-if is trivially checked=True/improved=True
        (nothing to verify), so it always auto-applies -- confirmed end to end, not
        just asserted from what_if.py's own docstring."""
        registry = PlaybookRegistry()
        chain = ProviderChain([])
        graph = build_supervisor_graph(chain, registry, DUMMY_PATH, DUMMY_PATH, InMemorySaver())

        result = graph.invoke(
            initial_state(_incident("emergency_run")),
            config={"configurable": {"thread_id": "t8"}},
        )

        assert "__interrupt__" not in result
        assert result["proposal"]["playbook_name"] == "hold_current_plan"
        assert result["status"] == "cleared"


class TestAutoApplyEligibility:
    """A playbook that was simulated AND improved things only auto-applies if it is one
    of the low-impact ones (a controller-mode change or a no-op). Rerouting traffic can
    be shown to help, but it still goes to a human -- with the evidence attached."""

    def test_checked_and_improved_reroute_still_asks_a_human(self) -> None:
        patcher, graph = _graph_with_fake_what_if(
            WhatIfResult(
                checked=True,
                improved=True,
                baseline_mean_pressure=70.0,
                proposed_mean_pressure=55.0,
                metric="mean_halted_vehicles",
            )
        )
        with patcher:
            paused = graph.invoke(
                initial_state(_incident("accident")),
                config={"configurable": {"thread_id": "t-elig-1"}},
            )

        assert paused["proposal"]["playbook_name"] == "divert_arterial"
        assert paused["auto_applied"] is False
        interrupt_payload = paused["__interrupt__"][0].value
        assert interrupt_payload["what_if_checked"] is True
        assert interrupt_payload["what_if_improved"] is True

    def test_the_simulated_evidence_is_carried_in_state_and_the_interrupt(self) -> None:
        patcher, graph = _graph_with_fake_what_if(
            WhatIfResult(
                checked=True,
                improved=True,
                baseline_mean_pressure=70.0,
                proposed_mean_pressure=55.0,
                metric="mean_halted_vehicles",
            )
        )
        expected = {"metric": "mean_halted_vehicles", "baseline": 70.0, "proposed": 55.0}
        with patcher:
            paused = graph.invoke(
                initial_state(_incident("accident")),
                config={"configurable": {"thread_id": "t-elig-2"}},
            )

        assert paused["what_if_detail"] == expected
        assert paused["__interrupt__"][0].value["what_if_detail"] == expected

    def test_unchecked_what_if_has_no_detail(self) -> None:
        patcher, graph = _graph_with_fake_what_if(
            WhatIfResult(
                checked=False,
                improved=None,
                baseline_mean_pressure=None,
                proposed_mean_pressure=None,
            )
        )
        with patcher:
            paused = graph.invoke(
                initial_state(_incident("weather")),
                config={"configurable": {"thread_id": "t-elig-3"}},
            )

        assert paused["what_if_detail"] is None

    def test_the_incident_is_handed_to_the_what_if_tool(self) -> None:
        registry = PlaybookRegistry()
        graph = build_supervisor_graph(
            ProviderChain([]), registry, DUMMY_PATH, DUMMY_PATH, InMemorySaver()
        )
        unchecked = WhatIfResult(
            checked=False, improved=None, baseline_mean_pressure=None, proposed_mean_pressure=None
        )
        with patch("atlas.cognitive.supervisor.run_what_if", return_value=unchecked) as mock:
            graph.invoke(
                initial_state(_incident("accident")),
                config={"configurable": {"thread_id": "t-elig-4"}},
            )

        assert mock.call_args.kwargs["incident"]["class"] == "accident"
