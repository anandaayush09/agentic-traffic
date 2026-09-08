"""Validates IntersectionAgentService end to end against the real sandbox network (fast,
already used by other tests) -- decide_and_step() actually stepping a live SUMO twin, the
FastAPI /health and /mode endpoints, and control.commands.v1/system.health.v1 messages
published through a fake bus producer (no broker needed, same Protocol-based fake used in
tests/unit/test_bus_producer_consumer.py).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from fastapi.testclient import TestClient

from atlas.bus.producer import BusProducer
from atlas.bus.schema_registry import SchemaRegistry
from atlas.common.config import project_root
from atlas.control.shield import ShieldConfig
from atlas.services.auth import create_operator_token
from atlas.services.intersection_agent import (
    IntersectionAgentConfig,
    IntersectionAgentService,
    create_app,
)

NETWORK_DIR = project_root() / "data" / "networks" / "sandbox"
TLS_ID = "B1"
JWT_SECRET = "test-secret-do-not-use-in-prod-32-bytes-min"


@dataclass
class _FakeRawProducer:
    sent: list[tuple[str, bytes | None, bytes | None]] = field(default_factory=list)

    def send(self, topic: str, value: bytes | None = None, key: bytes | None = None) -> object:
        self.sent.append((topic, value, key))
        return None

    def flush(self, timeout: float | None = None) -> None:
        pass


def _make_config() -> IntersectionAgentConfig:
    return IntersectionAgentConfig(
        tls_id=TLS_ID,
        net_file=NETWORK_DIR / "net.net.xml",
        route_file=NETWORK_DIR / "routes.rou.xml",
        max_lanes=4,
        max_green_phases=2,
        decision_interval_s=5.0,
        shield_config=ShieldConfig(min_green_s=10.0, max_green_s=60.0),
        default_mode="max_pressure",
    )


class TestDecideAndStep:
    def test_max_pressure_mode_publishes_valid_command_and_heartbeat(self) -> None:
        raw = _FakeRawProducer()
        bus = BusProducer(raw, SchemaRegistry())
        service = IntersectionAgentService(_make_config(), bus_producer=bus)
        service.start()
        try:
            info = service.decide_and_step()
        finally:
            service.stop()

        assert "pressure" in info
        topics = [topic for topic, _, _ in raw.sent]
        assert "control.commands.v1" in topics
        assert "system.health.v1" in topics

        command_value = next(v for t, v, _ in raw.sent if t == "control.commands.v1")
        command = json.loads(command_value)  # type: ignore[arg-type]
        assert command["intersection_id"] == TLS_ID
        assert command["actor"] == "max_pressure"

    def test_switching_to_fixed_time_mode_changes_actor_on_next_command(self) -> None:
        raw = _FakeRawProducer()
        bus = BusProducer(raw, SchemaRegistry())
        service = IntersectionAgentService(_make_config(), bus_producer=bus)
        service.mode = "fixed_time"
        service.start()
        try:
            service.decide_and_step()
        finally:
            service.stop()

        command_value = next(v for t, v, _ in raw.sent if t == "control.commands.v1")
        assert json.loads(command_value)["actor"] == "fixed_time"  # type: ignore[arg-type]

    def test_works_without_a_bus_producer(self) -> None:
        """A bus-less agent (no Redpanda reachable) must still run its control loop --
        publishing is best-effort, not load-bearing for the decision itself."""
        service = IntersectionAgentService(_make_config())
        service.start()
        try:
            info = service.decide_and_step()
        finally:
            service.stop()
        assert "pressure" in info


class TestIntersectionAgentApp:
    def test_health_and_mode_endpoints(self) -> None:
        service = IntersectionAgentService(_make_config())
        service.start()
        app = create_app(service, JWT_SECRET, start_background_loop=False)
        token = create_operator_token("operator-1", JWT_SECRET)
        try:
            with TestClient(app) as client:
                health = client.get("/health").json()
                assert health["tls_id"] == TLS_ID
                assert health["mode"] == "max_pressure"

                mode_response = client.post(
                    "/mode",
                    json={"mode": "fixed_time"},
                    headers={"Authorization": f"Bearer {token}"},
                )
                assert mode_response.status_code == 200
                assert mode_response.json() == {"mode": "fixed_time"}
                assert client.get("/mode").json() == {"mode": "fixed_time"}
        finally:
            service.stop()

    def test_invalid_mode_is_rejected(self) -> None:
        service = IntersectionAgentService(_make_config())
        service.start()
        app = create_app(service, JWT_SECRET, start_background_loop=False)
        token = create_operator_token("operator-1", JWT_SECRET)
        try:
            with TestClient(app) as client:
                response = client.post(
                    "/mode",
                    json={"mode": "not_a_real_mode"},
                    headers={"Authorization": f"Bearer {token}"},
                )
                assert response.status_code == 422
        finally:
            service.stop()

    def test_set_mode_without_a_token_is_rejected(self) -> None:
        """The real point of Week 10's fix -- POST /mode used to be unauthenticated
        entirely; confirms it now genuinely requires a valid operator JWT."""
        service = IntersectionAgentService(_make_config())
        service.start()
        app = create_app(service, JWT_SECRET, start_background_loop=False)
        try:
            with TestClient(app) as client:
                response = client.post("/mode", json={"mode": "fixed_time"})
                assert response.status_code == 401
                assert client.get("/mode").json() == {"mode": "max_pressure"}
        finally:
            service.stop()
