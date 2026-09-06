"""Validates the Watchdog's staleness-detection logic with a fake clock -- the actual
mechanism the Week 6 roadmap's "watchdog failover test (kill agent process, assert
fallback engages within 2 cycles)" chaos test needs to be right, tested deterministically
without real sleeps, a live broker, or an actual process kill.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient

from atlas.services.watchdog import Watchdog, create_app

T0 = datetime(2026, 1, 1, tzinfo=UTC)
HEARTBEAT_INTERVAL_S = 5.0
STALE_AFTER_S = 2 * HEARTBEAT_INTERVAL_S  # "after 2 missed cycles"


class TestRecordAndStale:
    def test_freshly_seen_service_is_not_stale(self) -> None:
        watchdog = Watchdog(stale_after_s=STALE_AFTER_S)
        watchdog.record_heartbeat("intersection_agent.42430333", T0)

        assert watchdog.stale_services(T0 + timedelta(seconds=1)) == []

    def test_never_seen_service_is_not_reported_stale(self) -> None:
        """Nothing to have gone quiet from yet -- a startup-ordering question, not a
        failure to detect."""
        watchdog = Watchdog(stale_after_s=STALE_AFTER_S)
        assert watchdog.stale_services(T0) == []

    def test_service_missing_more_than_2_cycles_is_stale(self) -> None:
        """The literal Week 6 chaos-test scenario: an agent stops heartbeating (as if
        its process died) and must be detected as stale within 2 missed cycles."""
        watchdog = Watchdog(stale_after_s=STALE_AFTER_S)
        watchdog.record_heartbeat("intersection_agent.42430333", T0)

        now = T0 + timedelta(seconds=STALE_AFTER_S + 0.1)

        assert watchdog.stale_services(now) == ["intersection_agent.42430333"]

    def test_service_missing_less_than_2_cycles_is_not_yet_stale(self) -> None:
        watchdog = Watchdog(stale_after_s=STALE_AFTER_S)
        watchdog.record_heartbeat("intersection_agent.42430333", T0)

        now = T0 + timedelta(seconds=STALE_AFTER_S - 0.1)

        assert watchdog.stale_services(now) == []

    def test_out_of_order_heartbeat_does_not_move_last_seen_backward(self) -> None:
        """At-least-once delivery (docs/design/05_communication.md section 5.6) can
        redeliver an old heartbeat after a fresher one already arrived -- must not make
        the service look staler than it actually is."""
        watchdog = Watchdog(stale_after_s=STALE_AFTER_S)
        watchdog.record_heartbeat("svc", T0 + timedelta(seconds=10))
        watchdog.record_heartbeat("svc", T0)  # late, out-of-order redelivery

        assert watchdog.stale_services(T0 + timedelta(seconds=11)) == []

    def test_ingest_parses_a_real_heartbeat_payload(self) -> None:
        watchdog = Watchdog(stale_after_s=STALE_AFTER_S)
        watchdog.ingest(
            {
                "msg_id": "x",
                "ts": T0.isoformat(),
                "schema_version": "1",
                "service_id": "intersection_agent.42430333",
                "status": "ok",
            }
        )

        assert watchdog.stale_services(T0) == []
        assert "intersection_agent.42430333" in watchdog.last_seen_snapshot()


class TestWatchdogApp:
    def test_health_endpoint(self) -> None:
        client = TestClient(create_app(Watchdog(stale_after_s=STALE_AFTER_S)))

        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_status_endpoint_reports_stale_services(self) -> None:
        watchdog = Watchdog(stale_after_s=0.0)  # anything unseen for >0s is stale
        watchdog.record_heartbeat("intersection_agent.42430333", T0)
        client = TestClient(create_app(watchdog))

        response = client.get("/status")

        assert response.status_code == 200
        body = response.json()
        assert "intersection_agent.42430333" in body["stale_services"]
        assert "intersection_agent.42430333" in body["last_seen"]
