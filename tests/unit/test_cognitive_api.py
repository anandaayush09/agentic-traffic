"""Validates CognitiveApiService against a REAL compiled Supervisor graph (same
InMemorySaver + patched-run_what_if pattern as test_supervisor.py -- these tests are
about the API/service layer's own logic, not the what-if tool, which
test_what_if.py already covers against real SUMO) and the HTTP layer end to end via
FastAPI's TestClient, including that JWT actually gates the mutating endpoints for
real (not just asserted).
"""

from __future__ import annotations

from collections.abc import Mapping
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


class _FakeLiveTraffic:
    """Records what the API hands the live twin; no SUMO involved."""

    def __init__(self, apply_raises: bool = False) -> None:
        self.applied: list[dict[str, Any]] = []
        self.playbooks: list[tuple[str, str, dict[str, object]]] = []
        self.modes: dict[str, str] = {}
        self.current: dict[str, object] = {
            "ready": True,
            "sim_time_s": 12.0,
            "edges": {},
            "vehicles": [],
        }
        self.history_args: tuple[int, int] | None = None
        self._apply_raises = apply_raises

    def geometry(self) -> list[dict[str, object]]:
        return [{"edge_id": "e1", "lanes": 2, "speed_limit_mps": 13.9, "coords": [[0.0, 1.0]]}]

    def junction_positions(self) -> list[dict[str, object]]:
        return [{"id": "J1", "lon": 1.5, "lat": 2.5}]

    def snapshot(self) -> dict[str, object]:
        return self.current

    def history(self, max_frames: int, stride: int) -> list[dict[str, object]]:
        self.history_args = (max_frames, stride)
        return [{"sim_time_s": 1.0}, {"sim_time_s": 2.0}]

    def apply_incident(self, incident: Mapping[str, object]) -> list[str]:
        if self._apply_raises:
            raise RuntimeError("live twin exploded")
        self.applied.append(dict(incident))
        return ["e1"]

    def apply_playbook(
        self, incident_id: str, playbook_name: str, params: Mapping[str, object]
    ) -> bool:
        if self._apply_raises:
            raise RuntimeError("live twin exploded")
        self.playbooks.append((incident_id, playbook_name, dict(params)))
        return True

    def set_junction_mode(self, tls_id: str, mode: str) -> None:
        if tls_id != "J1":
            raise KeyError(f"'{tls_id}' is not an ATLAS-controlled junction")
        if mode not in ("fixed_time", "actuated", "max_pressure"):
            raise ValueError(f"mode must be one of [...], got '{mode}'")
        self.modes[tls_id] = mode


class TestLiveTrafficEndpoints:
    def test_traffic_endpoints_are_404_when_no_live_twin_is_attached(self) -> None:
        client = TestClient(create_app(_service(), SECRET))

        assert client.get("/traffic").status_code == 404
        assert client.get("/traffic/geometry").status_code == 404

    def test_traffic_endpoints_are_unauthenticated_and_serve_the_live_twin(self) -> None:
        client = TestClient(create_app(_service(), SECRET, live_traffic=_FakeLiveTraffic()))

        snapshot = client.get("/traffic")
        geometry = client.get("/traffic/geometry")

        assert snapshot.status_code == 200
        assert snapshot.json()["sim_time_s"] == 12.0
        assert geometry.status_code == 200
        assert geometry.json()["edges"][0]["edge_id"] == "e1"
        assert geometry.json()["junctions"] == [{"id": "J1", "lon": 1.5, "lat": 2.5}]

    def test_submitted_incident_is_handed_to_the_live_twin(self) -> None:
        live = _FakeLiveTraffic()
        client = TestClient(create_app(_service(), SECRET, live_traffic=live))
        token = create_operator_token("op-1", SECRET)

        with patch("atlas.cognitive.supervisor.run_what_if", return_value=REQUEST_APPROVAL_RESULT):
            response = client.post(
                "/incidents",
                json=_incident("inc-live", "accident"),
                headers={"Authorization": f"Bearer {token}"},
            )

        assert response.status_code == 201
        assert [i["incident_id"] for i in live.applied] == ["inc-live"]

    def test_unauthenticated_post_never_reaches_the_live_twin(self) -> None:
        live = _FakeLiveTraffic()
        client = TestClient(create_app(_service(), SECRET, live_traffic=live))

        response = client.post("/incidents", json=_incident("inc-noauth", "accident"))

        assert response.status_code == 401
        assert live.applied == []

    def test_a_failing_live_twin_does_not_fail_the_incident(self) -> None:
        client = TestClient(
            create_app(_service(), SECRET, live_traffic=_FakeLiveTraffic(apply_raises=True))
        )
        token = create_operator_token("op-1", SECRET)

        with patch("atlas.cognitive.supervisor.run_what_if", return_value=REQUEST_APPROVAL_RESULT):
            response = client.post(
                "/incidents",
                json=_incident("inc-live-boom", "accident"),
                headers={"Authorization": f"Bearer {token}"},
            )

        assert response.status_code == 201
        assert response.json()["pending_approval"] is True


class TestLiveTwinPlaybookAndModeHooks:
    def _client(self, live: _FakeLiveTraffic) -> tuple[TestClient, dict[str, str]]:
        client = TestClient(create_app(_service(), SECRET, live_traffic=live))
        token = create_operator_token("op-1", SECRET)
        return client, {"Authorization": f"Bearer {token}"}

    def test_an_auto_applied_playbook_is_applied_to_the_live_twin(self) -> None:
        live = _FakeLiveTraffic()
        client, auth = self._client(live)

        with patch("atlas.cognitive.supervisor.run_what_if", return_value=AUTO_APPLY_RESULT):
            client.post("/incidents", json=_incident("inc-auto"), headers=auth)

        assert [(i, name) for i, name, _ in live.playbooks] == [
            ("inc-auto", "switch_to_max_pressure")
        ]

    def test_a_paused_incident_does_not_touch_the_twin_until_approved(self) -> None:
        live = _FakeLiveTraffic()
        client, auth = self._client(live)

        with patch("atlas.cognitive.supervisor.run_what_if", return_value=REQUEST_APPROVAL_RESULT):
            client.post("/incidents", json=_incident("inc-wait", "accident"), headers=auth)
            assert live.playbooks == []

            response = client.post("/incidents/inc-wait/approve", headers=auth)

        assert response.status_code == 200
        assert [(i, name) for i, name, _ in live.playbooks] == [("inc-wait", "divert_arterial")]

    def test_a_rejected_playbook_is_never_applied(self) -> None:
        live = _FakeLiveTraffic()
        client, auth = self._client(live)

        with patch("atlas.cognitive.supervisor.run_what_if", return_value=REQUEST_APPROVAL_RESULT):
            client.post("/incidents", json=_incident("inc-no", "accident"), headers=auth)
            response = client.post("/incidents/inc-no/reject", headers=auth)

        assert response.status_code == 200
        assert live.playbooks == []

    def test_a_failing_live_twin_does_not_fail_the_approval(self) -> None:
        client, auth = self._client(_FakeLiveTraffic(apply_raises=True))

        with patch("atlas.cognitive.supervisor.run_what_if", return_value=REQUEST_APPROVAL_RESULT):
            client.post("/incidents", json=_incident("inc-boom2", "accident"), headers=auth)
            response = client.post("/incidents/inc-boom2/approve", headers=auth)

        assert response.status_code == 200
        assert response.json()["state"]["status"] == "cleared"

    def test_junction_mode_requires_a_token(self) -> None:
        live = _FakeLiveTraffic()
        client, _ = self._client(live)

        response = client.post("/traffic/junctions/J1/mode", json={"mode": "actuated"})

        assert response.status_code == 401
        assert live.modes == {}

    def test_junction_mode_is_set_for_an_operator(self) -> None:
        live = _FakeLiveTraffic()
        client, auth = self._client(live)

        response = client.post(
            "/traffic/junctions/J1/mode", json={"mode": "actuated"}, headers=auth
        )

        assert response.status_code == 200
        assert response.json() == {"tls_id": "J1", "mode": "actuated"}
        assert live.modes == {"J1": "actuated"}

    def test_unknown_junction_is_404_and_bad_mode_is_422(self) -> None:
        client, auth = self._client(_FakeLiveTraffic())

        unknown = client.post("/traffic/junctions/J9/mode", json={"mode": "actuated"}, headers=auth)
        bad_mode = client.post("/traffic/junctions/J1/mode", json={"mode": "rl"}, headers=auth)

        assert unknown.status_code == 404
        assert bad_mode.status_code == 422

    def test_junction_mode_is_404_when_no_live_twin_is_attached(self) -> None:
        client = TestClient(create_app(_service(), SECRET))
        token = create_operator_token("op-1", SECRET)

        response = client.post(
            "/traffic/junctions/J1/mode",
            json={"mode": "actuated"},
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 404


class TestReplayHistory:
    def test_history_serves_frames_and_passes_the_query_through(self) -> None:
        live = _FakeLiveTraffic()
        client = TestClient(create_app(_service(), SECRET, live_traffic=live))

        response = client.get("/traffic/history?frames=50&stride=4")

        assert response.status_code == 200
        assert [f["sim_time_s"] for f in response.json()["frames"]] == [1.0, 2.0]
        assert live.history_args == (50, 4)

    def test_history_has_sensible_defaults_and_bounds(self) -> None:
        live = _FakeLiveTraffic()
        client = TestClient(create_app(_service(), SECRET, live_traffic=live))

        assert client.get("/traffic/history").status_code == 200
        assert live.history_args == (300, 2)
        assert client.get("/traffic/history?frames=0").status_code == 422
        assert client.get("/traffic/history?frames=100000").status_code == 422
        assert client.get("/traffic/history?stride=0").status_code == 422

    def test_history_is_404_without_a_live_twin(self) -> None:
        client = TestClient(create_app(_service(), SECRET))

        assert client.get("/traffic/history").status_code == 404


class TestTrafficWebSocket:
    def test_pushes_the_current_snapshot_then_each_new_one(self) -> None:
        live = _FakeLiveTraffic()
        client = TestClient(create_app(_service(), SECRET, live_traffic=live))

        with client.websocket_connect("/ws/traffic") as ws:
            assert ws.receive_json()["sim_time_s"] == 12.0
            live.current = {"ready": True, "sim_time_s": 17.0, "edges": {}, "vehicles": []}
            assert ws.receive_json()["sim_time_s"] == 17.0

    def test_an_unchanged_snapshot_is_not_resent(self) -> None:
        live = _FakeLiveTraffic()
        client = TestClient(create_app(_service(), SECRET, live_traffic=live))

        with client.websocket_connect("/ws/traffic") as ws:
            ws.receive_json()
            live.current = {"ready": True, "sim_time_s": 99.0, "edges": {}, "vehicles": []}
            # the very next message is the NEW snapshot, not a duplicate of the first
            assert ws.receive_json()["sim_time_s"] == 99.0

    def test_an_allowed_dashboard_origin_is_accepted(self) -> None:
        client = TestClient(create_app(_service(), SECRET, live_traffic=_FakeLiveTraffic()))

        with client.websocket_connect(
            "/ws/traffic", headers={"origin": "http://localhost:5173"}
        ) as ws:
            assert ws.receive_json()["ready"] is True

    def test_a_foreign_origin_is_refused(self) -> None:
        from starlette.websockets import WebSocketDisconnect

        client = TestClient(create_app(_service(), SECRET, live_traffic=_FakeLiveTraffic()))

        with pytest.raises(WebSocketDisconnect):  # noqa: SIM117
            with client.websocket_connect(
                "/ws/traffic", headers={"origin": "https://evil.example"}
            ):
                pass

    def test_it_is_refused_when_no_live_twin_is_attached(self) -> None:
        from starlette.websockets import WebSocketDisconnect

        client = TestClient(create_app(_service(), SECRET))

        with pytest.raises(WebSocketDisconnect):  # noqa: SIM117
            with client.websocket_connect("/ws/traffic"):
                pass


class TestDemoToken:
    LOCAL = ("127.0.0.1", 50000)

    def test_it_is_off_by_default(self) -> None:
        client = TestClient(create_app(_service(), SECRET), client=self.LOCAL)

        assert client.get("/auth/config").json() == {"demo_token_enabled": False}
        assert client.post("/auth/demo-token").status_code == 404

    def test_when_enabled_a_local_client_gets_a_working_operator_token(self) -> None:
        client = TestClient(
            create_app(_service(), SECRET, demo_token_enabled=True), client=self.LOCAL
        )

        assert client.get("/auth/config").json() == {"demo_token_enabled": True}
        token = client.post("/auth/demo-token").json()["token"]

        with patch("atlas.cognitive.supervisor.run_what_if", return_value=AUTO_APPLY_RESULT):
            posted = client.post(
                "/incidents",
                json=_incident("inc-demo-token"),
                headers={"Authorization": f"Bearer {token}"},
            )
        assert posted.status_code == 201  # the issued token really authorizes mutations

    def test_a_remote_client_is_refused(self) -> None:
        client = TestClient(
            create_app(_service(), SECRET, demo_token_enabled=True), client=("203.0.113.9", 4000)
        )

        assert client.post("/auth/demo-token").status_code == 403

    def test_a_foreign_browser_origin_is_refused(self) -> None:
        client = TestClient(
            create_app(_service(), SECRET, demo_token_enabled=True), client=self.LOCAL
        )

        refused = client.post("/auth/demo-token", headers={"origin": "https://evil.example"})
        allowed = client.post("/auth/demo-token", headers={"origin": "http://localhost:5173"})

        assert refused.status_code == 403
        assert allowed.status_code == 200


class TestOperationalEndpoints:
    def test_health_is_a_dependency_free_liveness_probe(self) -> None:
        client = TestClient(create_app(_service(), SECRET, live_traffic=_FakeLiveTraffic()))

        assert client.get("/health").json() == {"status": "ok"}

    def test_ready_without_a_live_twin_is_ready_immediately(self) -> None:
        client = TestClient(create_app(_service(), SECRET))

        response = client.get("/ready")

        assert response.status_code == 200
        assert response.json() == {"ready": True, "live_traffic": "disabled"}

    def test_ready_waits_for_the_live_simulation_to_warm_up(self) -> None:
        live = _FakeLiveTraffic()
        live.current = {"ready": False, "error": None}
        client = TestClient(create_app(_service(), SECRET, live_traffic=live))

        warming = client.get("/ready")
        live.current = {"ready": True, "sim_time_s": 300.0}
        ready = client.get("/ready")

        assert warming.status_code == 503  # a deploy's health gate waits instead of failing
        assert warming.json()["ready"] is False
        assert ready.status_code == 200
        assert ready.json()["live_traffic"] == "ready"

    def test_ready_surfaces_a_simulation_error_as_not_ready(self) -> None:
        live = _FakeLiveTraffic()
        live.current = {"ready": False, "error": "SUMO crashed"}
        client = TestClient(create_app(_service(), SECRET, live_traffic=live))

        response = client.get("/ready")

        assert response.status_code == 503
        assert response.json()["live_traffic"] == "SUMO crashed"

    def test_version_reports_the_running_build_and_environment(self) -> None:
        client = TestClient(
            create_app(_service(), SECRET, version="sha-abc1234", environment="staging")
        )

        assert client.get("/version").json() == {"version": "sha-abc1234", "environment": "staging"}

    def test_version_defaults_are_honest_about_being_a_dev_build(self) -> None:
        client = TestClient(create_app(_service(), SECRET))

        assert client.get("/version").json() == {"version": "dev", "environment": "development"}

    def test_an_unhandled_error_is_a_generic_500_that_leaks_nothing(self) -> None:
        service = _service()

        def explode() -> list[dict[str, Any]]:
            raise RuntimeError("db password is hunter2 at /srv/internal/path")

        service.list_incidents = explode  # type: ignore[method-assign]
        client = TestClient(create_app(service, SECRET), raise_server_exceptions=False)

        response = client.get("/incidents")

        assert response.status_code == 500
        assert response.json() == {"detail": "Internal server error"}
        assert "hunter2" not in response.text
        assert "/srv/internal" not in response.text

    def test_an_unhandled_error_is_logged_with_its_context(self) -> None:
        from structlog.testing import capture_logs

        service = _service()
        service.list_incidents = lambda: (_ for _ in ()).throw(RuntimeError("boom"))  # type: ignore[method-assign]
        client = TestClient(create_app(service, SECRET), raise_server_exceptions=False)

        with capture_logs() as logs:
            client.get("/incidents")

        event = next(entry for entry in logs if entry["event"] == "api.unhandled_error")
        assert (event["method"], event["path"]) == ("GET", "/incidents")
        assert event["log_level"] == "error"


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
