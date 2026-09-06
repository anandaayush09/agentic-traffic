"""Watchdog: consumes `system.health.v1` heartbeats and detects stale services -- docs/
design/04_agent_architecture.md section 4.1's failure-recovery pattern ("if the agent
process dies, its Redis heartbeat key expires... after 2 missed cycles, the junction's
local fallback... takes over") and the Week 6 roadmap's "watchdog failover test (kill
agent process, assert fallback engages within 2 cycles) -- first chaos test".

Redis isn't wired up yet (still Week 5-6/later control-plane work per the ATLAS dev-
environment memory), so staleness here is tracked from the bus heartbeat stream directly
rather than via Redis TTL'd keys -- same detection semantics (an expired/missing recent
heartbeat means the service is presumed dead), different storage, and a documented
substitution rather than a silent gap: swapping in a Redis-backed `last_seen` store later
is a storage change, not a redesign of `stale_services`'s logic.

`Watchdog` itself is pure clock/dict logic (`record_heartbeat`/`stale_services`),
deliberately separated from the bus-consuming loop, so the actual staleness-detection
algorithm the Week 6 chaos test needs to verify is testable with a fake clock and no
broker at all.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI

from atlas.bus.consumer import BusConsumer
from atlas.common.logging import get_logger

logger = get_logger(__name__)


class Watchdog:
    def __init__(self, stale_after_s: float) -> None:
        self._stale_after_s = stale_after_s
        self._last_seen: dict[str, datetime] = {}

    def record_heartbeat(self, service_id: str, ts: datetime) -> None:
        """Only advances `last_seen` forward -- an out-of-order/late heartbeat (at-
        least-once delivery, docs/design/05_communication.md section 5.6) must never
        make a service look MORE stale than the freshest heartbeat already says."""
        current = self._last_seen.get(service_id)
        if current is None or ts > current:
            self._last_seen[service_id] = ts

    def stale_services(self, now: datetime) -> list[str]:
        """Services that HAVE sent at least one heartbeat but not within
        `stale_after_s` of `now`. A service never seen at all is not "stale" -- there is
        nothing to have gone quiet from yet; that is a startup-ordering question, not a
        failure to detect.
        """
        return [
            service_id
            for service_id, last_seen in self._last_seen.items()
            if (now - last_seen).total_seconds() > self._stale_after_s
        ]

    def last_seen_snapshot(self) -> dict[str, str]:
        return {service_id: ts.isoformat() for service_id, ts in self._last_seen.items()}

    def ingest(self, payload: dict[str, Any]) -> None:
        service_id = payload["service_id"]
        ts = datetime.fromisoformat(payload["ts"])
        self.record_heartbeat(service_id, ts)

    def consume_forever(self, consumer: BusConsumer) -> None:
        """Blocking loop -- run this in a background thread (`asyncio.to_thread`) from
        an async service, since kafka-python's consumer iteration is itself blocking.
        """
        for payload in consumer.consume():
            self.ingest(payload)


def create_app(watchdog: Watchdog, consumer: BusConsumer | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # kafka-python's consumer iteration is blocking, so it runs in a worker thread
        # (asyncio.to_thread) rather than blocking the event loop that serves /health and
        # /status. Known limitation: task.cancel() does not interrupt a running thread,
        # so this loop keeps consuming until the process itself exits -- acceptable for
        # a service meant to run for the container's lifetime, not something this
        # shutdown path currently handles gracefully.
        task = (
            asyncio.create_task(asyncio.to_thread(watchdog.consume_forever, consumer))
            if consumer is not None
            else None
        )
        try:
            yield
        finally:
            if task is not None:
                task.cancel()

    app = FastAPI(title="Watchdog", lifespan=lifespan)

    @app.get("/health")
    def get_health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/status")
    def get_status() -> dict[str, Any]:
        now = datetime.now(UTC)
        return {
            "last_seen": watchdog.last_seen_snapshot(),
            "stale_services": watchdog.stale_services(now),
        }

    return app


def main() -> None:
    """CLI entrypoint: `python -m atlas.services.watchdog --kafka-brokers localhost:9092
    [--stale-after-s 10]`. Used directly and by docker-compose's watchdog service.
    """
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--kafka-brokers", required=True)
    parser.add_argument("--stale-after-s", type=float, default=10.0)
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    watchdog = Watchdog(stale_after_s=args.stale_after_s)
    consumer = BusConsumer.connect([args.kafka_brokers], "system.health.v1", group_id="watchdog")
    uvicorn.run(create_app(watchdog, consumer), host="0.0.0.0", port=args.port)  # noqa: S104


if __name__ == "__main__":
    main()
