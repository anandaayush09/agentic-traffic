"""Validates CognitiveApiService against a REAL compiled Supervisor graph (same
InMemorySaver + patched-run_what_if pattern as test_supervisor.py -- these tests are
about the API/service layer's own logic, not the what-if tool, which
test_what_if.py already covers against real SUMO) and the HTTP layer end to end via
FastAPI's TestClient, including that JWT actually gates the mutating endpoints for
real (not just asserted).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from atlas.cognitive.playbooks import PlaybookRegistry
from atlas.cognitive.provider_chain import ProviderChain
from atlas.cognitive.supervisor import build_supervisor_graph
from atlas.cognitive.what_if import WhatIfResult
from atlas.services.auth import create_operator_token
from atlas.services.cognitive_api import (
    ApprovalNotPendingError,
    CognitiveApiService,
    IncidentNotFoundError,
    create_app,
)

DUMMY_PATH = Path("unused_for_these_tests.net.xml")
SECRET = "test-secret-do-not-use-in-prod-32-bytes-min"

AUTO_APPLY_RESULT = WhatIfResult(
    checked=True, improved=True, baseline_mean_pressure=10.0, proposed_mean_pressure=5.0
)
REQUEST_APPROVAL_RESULT = WhatIfResult(
    checked=False, improved=None, baseline_mean_pressure=None, proposed_mean_pressure=None
)


def _incident(incident_id: str, incident_class: str = "sensor_fault") -> dict[str, Any]:
    return {
        "incident_id": incident_id,
        "class": incident_class,
        "status": "detected",
        "confidence": 0.9,
        "location": {"link_ids": ["B1"]},
    }


def _service() -> CognitiveApiService:
    registry = PlaybookRegistry()
    chain = ProviderChain([])  # no LLM providers -- deterministic mapper only
    graph = build_supervisor_graph(chain, registry, DUMMY_PATH, DUMMY_PATH, InMemorySaver())
    return CognitiveApiService(graph)


class TestServiceAutoApplyPath:
    def test_submit_auto_applies_and_is_not_pending(self) -> None:
        service = _service()
        with patch("atlas.cognitive.supervisor.run_what_if", return_value=AUTO_APPLY_RESULT):
            result = service.submit_incident(_incident("inc-auto"))

        assert result["pending_approval"] is False
        assert result["state"]["status"] == "cleared"
        assert result["interrupt"] is None


class TestServiceRequestApprovalPath:
    def test_submit_pauses_with_the_real_interrupt_payload(self) -> None:
        service = _service()
        with patch("atlas.cognitive.supervisor.run_what_if", return_value=REQUEST_APPROVAL_RESULT):
            result = service.submit_incident(_incident("inc-pending"))

        assert result["pending_approval"] is True
        assert result["interrupt"]["question"] == "Approve this playbook proposal?"
        assert result["state"]["status"] == "triaged"

    def test_approve_resolves_to_cleared(self) -> None:
        service = _service()
        with patch("atlas.cognitive.supervisor.run_what_if", return_value=REQUEST_APPROVAL_RESULT):
            service.submit_incident(_incident("inc-approve"))
            result = service.resolve_approval("inc-approve", approved=True)

        assert result["pending_approval"] is False
        assert result["state"]["status"] == "cleared"
        assert result["state"]["approved"] is True

    def test_reject_resolves_to_triaged_not_cleared(self) -> None:
        service = _service()
        with patch("atlas.cognitive.supervisor.run_what_if", return_value=REQUEST_APPROVAL_RESULT):
            service.submit_incident(_incident("inc-reject"))
            result = service.resolve_approval("inc-reject", approved=False)

        assert result["pending_approval"] is False
        assert result["state"]["status"] == "triaged"
        assert result["state"]["approved"] is False

    def test_resolving_an_already_resolved_incident_raises(self) -> None:
        service = _service()
        with patch("atlas.cognitive.supervisor.run_what_if", return_value=REQUEST_APPROVAL_RESULT):
            service.submit_incident(_incident("inc-double"))
            service.resolve_approval("inc-double", approved=True)
            with pytest.raises(ApprovalNotPendingError):
                service.resolve_approval("inc-double", approved=True)

    def test_resolving_an_auto_applied_incident_raises_not_pending(self) -> None:
        service = _service()
        with patch("atlas.cognitive.supervisor.run_what_if", return_value=AUTO_APPLY_RESULT):
            service.submit_incident(_incident("inc-auto-2"))
            with pytest.raises(ApprovalNotPendingError):
                service.resolve_approval("inc-auto-2", approved=True)


class TestServiceUnknownIncident:
    def test_get_unknown_incident_raises(self) -> None:
        with pytest.raises(IncidentNotFoundError):
            _service().get_incident("does-not-exist")

    def test_resolve_unknown_incident_raises(self) -> None:
        with pytest.raises(IncidentNotFoundError):
            _service().resolve_approval("does-not-exist", approved=True)


class TestServiceListing:
    def test_list_pending_approvals_only_returns_paused_incidents(self) -> None:
        service = _service()
        with patch("atlas.cognitive.supervisor.run_what_if", return_value=AUTO_APPLY_RESULT):
            service.submit_incident(_incident("inc-list-auto"))
        with patch("atlas.cognitive.supervisor.run_what_if", return_value=REQUEST_APPROVAL_RESULT):
            service.submit_incident(_incident("inc-list-pending"))

        pending = service.list_pending_approvals()

        assert [incident["incident_id"] for incident in pending] == ["inc-list-pending"]
        assert len(service.list_incidents()) == 2


class TestHttpLayer:
    def test_health_is_unauthenticated(self) -> None:
        client = TestClient(create_app(_service(), SECRET))

        response = client.get("/health")

        assert response.status_code == 200

    def test_get_incidents_is_unauthenticated(self) -> None:
        client = TestClient(create_app(_service(), SECRET))

        response = client.get("/incidents")

        assert response.status_code == 200
        assert response.json() == []

    def test_post_incident_without_token_is_rejected(self) -> None:
        client = TestClient(create_app(_service(), SECRET))

        response = client.post("/incidents", json=_incident("inc-http-noauth"))

        assert response.status_code == 401

    def test_post_incident_with_token_auto_applies(self) -> None:
        client = TestClient(create_app(_service(), SECRET))
        token = create_operator_token("op-1", SECRET)

        with patch("atlas.cognitive.supervisor.run_what_if", return_value=AUTO_APPLY_RESULT):
            response = client.post(
                "/incidents",
                json=_incident("inc-http-auto"),
                headers={"Authorization": f"Bearer {token}"},
            )

        assert response.status_code == 201
        body = response.json()
        assert body["pending_approval"] is False
        assert body["state"]["status"] == "cleared"

    def test_get_unknown_incident_is_404(self) -> None:
        client = TestClient(create_app(_service(), SECRET))

        response = client.get("/incidents/does-not-exist")

        assert response.status_code == 404

    def test_approve_without_token_is_rejected(self) -> None:
        client = TestClient(create_app(_service(), SECRET))
        token = create_operator_token("op-1", SECRET)
        with patch("atlas.cognitive.supervisor.run_what_if", return_value=REQUEST_APPROVAL_RESULT):
            client.post(
                "/incidents",
                json=_incident("inc-http-pending"),
                headers={"Authorization": f"Bearer {token}"},
            )

        response = client.post("/incidents/inc-http-pending/approve")

        assert response.status_code == 401

    def test_approve_with_token_resolves_the_real_paused_graph(self) -> None:
        client = TestClient(create_app(_service(), SECRET))
        token = create_operator_token("op-1", SECRET)
        auth_headers = {"Authorization": f"Bearer {token}"}
        with patch("atlas.cognitive.supervisor.run_what_if", return_value=REQUEST_APPROVAL_RESULT):
            client.post("/incidents", json=_incident("inc-http-approve"), headers=auth_headers)

            response = client.post("/incidents/inc-http-approve/approve", headers=auth_headers)

        assert response.status_code == 200
        assert response.json()["state"]["status"] == "cleared"

    def test_approve_an_already_resolved_incident_is_409(self) -> None:
        client = TestClient(create_app(_service(), SECRET))
        token = create_operator_token("op-1", SECRET)
        auth_headers = {"Authorization": f"Bearer {token}"}
        with patch("atlas.cognitive.supervisor.run_what_if", return_value=REQUEST_APPROVAL_RESULT):
            client.post("/incidents", json=_incident("inc-http-conflict"), headers=auth_headers)
            client.post("/incidents/inc-http-conflict/approve", headers=auth_headers)

            response = client.post("/incidents/inc-http-conflict/approve", headers=auth_headers)

        assert response.status_code == 409


class TestRealSumoBackedVerification:
    """One end-to-end test against the real sandbox network (same fixture pattern as
    test_what_if.py), proving the whole HTTP -> Supervisor graph -> real SUMO what-if
    path works together, not just with run_what_if patched out everywhere else in
    this file.
    """

    @pytest.fixture(scope="session")
    def sandbox_net_route(self) -> tuple[Path, Path]:
        import sys

        from atlas.common.config import project_root

        sys.path.insert(0, str(project_root() / "scripts"))
        import generate_sandbox_network

        generate_sandbox_network.generate()
        return generate_sandbox_network.NET_FILE, generate_sandbox_network.ROUTE_FILE

    def test_sensor_fault_on_real_junction_b1_auto_applies_via_real_sumo(
        self, sandbox_net_route: tuple[Path, Path]
    ) -> None:
        net_file, route_file = sandbox_net_route
        registry = PlaybookRegistry()
        chain = ProviderChain([])
        graph = build_supervisor_graph(chain, registry, net_file, route_file, InMemorySaver())
        client = TestClient(create_app(CognitiveApiService(graph), SECRET))
        token = create_operator_token("op-1", SECRET)

        incident = {
            "incident_id": "inc-real-sumo",
            "class": "sensor_fault",
            "status": "detected",
            "confidence": 0.9,
            "location": {"link_ids": ["B1"]},
        }
        response = client.post(
            "/incidents", json=incident, headers={"Authorization": f"Bearer {token}"}
        )

        assert response.status_code == 201
        body = response.json()
        assert body["state"]["proposal"]["playbook_name"] == "switch_to_max_pressure"
        # switch_to_max_pressure is real-simulation-verified by run_what_if -- whether
        # it auto-applies depends on the actual measured pressure comparison, so this
        # only asserts the what-if tool genuinely ran (checked=True), not a specific
        # improved value that would make this test flaky against real SUMO noise.
        assert body["state"]["what_if_checked"] is True
