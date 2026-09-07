"""Validates CorridorCoordinatorService against the real manhattan_midtown corridor
(the same 5 junctions Week 5's training uses) -- recompute() actually stepping a live
SUMO twin, publishing corridor.hints.v1/corridor.alerts.v1 through a fake bus producer
(no broker needed), and the /health and /corridor/{id}/state endpoints.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from fastapi.testclient import TestClient

from atlas.bus.producer import BusProducer
from atlas.bus.schema_registry import SchemaRegistry
from atlas.common.config import project_root
from atlas.services.corridor_coordinator import (
    CorridorCoordinatorConfig,
    CorridorCoordinatorService,
    create_app,
)

NETWORK_DIR = project_root() / "data" / "networks" / "manhattan_midtown"
CORRIDOR_TLS_IDS = ["42430333", "42430338", "42430342", "42430344", "42430347"]


@dataclass
class _FakeRawProducer:
    sent: list[tuple[str, bytes | None, bytes | None]] = field(default_factory=list)

    def send(self, topic: str, value: bytes | None = None, key: bytes | None = None) -> object:
        self.sent.append((topic, value, key))
        return None

    def flush(self, timeout: float | None = None) -> None:
        pass


def _make_config(**overrides: object) -> CorridorCoordinatorConfig:
    defaults: dict[str, object] = {
        "corridor_id": "test_corridor",
        "tls_ids": CORRIDOR_TLS_IDS,
        "net_file": NETWORK_DIR / "net.net.xml",
        "route_file": NETWORK_DIR / "demand" / "pm_peak.rou.xml",
        "additional_files": [NETWORK_DIR / "vtypes.add.xml"],
        "recompute_interval_s": 30.0,
    }
    defaults.update(overrides)
    return CorridorCoordinatorConfig(**defaults)  # type: ignore[arg-type]


class TestRecompute:
    def test_returns_one_hint_per_member_junction(self) -> None:
        service = CorridorCoordinatorService(_make_config())
        service.start()
        try:
            member_hints, _ = service.recompute()
        finally:
            service.stop()

        assert {h["intersection_id"] for h in member_hints} == set(CORRIDOR_TLS_IDS)
        # The first junction along the corridor is the offset reference point.
        first = next(h for h in member_hints if h["intersection_id"] == CORRIDOR_TLS_IDS[0])
        assert first["offset_s"] == 0.0

    def test_offsets_increase_monotonically_along_the_corridor(self) -> None:
        """Every consecutive segment has a positive real distance (verified in
        test_corridor_geometry.py), so cumulative offsets must strictly increase."""
        service = CorridorCoordinatorService(_make_config())
        service.start()
        try:
            member_hints, _ = service.recompute()
        finally:
            service.stop()

        offsets_in_order = [
            next(h["offset_s"] for h in member_hints if h["intersection_id"] == tls_id)
            for tls_id in CORRIDOR_TLS_IDS
        ]
        assert offsets_in_order == sorted(offsets_in_order)
        assert offsets_in_order[0] < offsets_in_order[-1]

    def test_publishes_valid_corridor_hints_message(self) -> None:
        raw = _FakeRawProducer()
        bus = BusProducer(raw, SchemaRegistry())
        service = CorridorCoordinatorService(_make_config(), bus_producer=bus)
        service.start()
        try:
            service.recompute()
        finally:
            service.stop()

        hints_value = next(v for t, v, _ in raw.sent if t == "corridor.hints.v1")
        message = json.loads(hints_value)  # type: ignore[arg-type]
        assert message["corridor_id"] == "test_corridor"
        assert len(message["member_hints"]) == len(CORRIDOR_TLS_IDS)

    def test_works_without_a_bus_producer(self) -> None:
        service = CorridorCoordinatorService(_make_config())
        service.start()
        try:
            member_hints, alerts = service.recompute()
        finally:
            service.stop()
        assert len(member_hints) == len(CORRIDOR_TLS_IDS)
        assert isinstance(alerts, list)


class TestCorridorCoordinatorApp:
    def test_health_endpoint(self) -> None:
        client = TestClient(create_app({}))
        assert client.get("/health").json() == {"status": "ok"}

    def test_state_endpoint_reflects_last_recompute(self) -> None:
        service = CorridorCoordinatorService(_make_config())
        service.start()
        try:
            service.recompute()
            client = TestClient(create_app({"test_corridor": service}))

            response = client.get("/corridor/test_corridor/state")

            assert response.status_code == 200
            body = response.json()
            assert body["corridor_id"] == "test_corridor"
            assert len(body["member_hints"]) == len(CORRIDOR_TLS_IDS)
        finally:
            service.stop()

    def test_unknown_corridor_returns_404(self) -> None:
        client = TestClient(create_app({}))
        response = client.get("/corridor/does_not_exist/state")
        assert response.status_code == 404
