"""Corridor Coordinator: computes green-wave offset hints and spillback alerts for one
arterial corridor every `recompute_interval_s` (docs/design/04_agent_architecture.md
section 4.2's "rolling-horizon offset recomputation every 60 s"), publishing
`corridor.hints.v1`/`corridor.alerts.v1` to the bus.

Owns a plain `SumoTwinEnv` (Week 1's primitive), NOT `MultiAgentSignalEnv` -- the
coordinator only ever READS state (platoon speeds, exit-link occupancy); it must never
become a second writer of any junction's signal (single-writer principle, docs/design/
05_communication.md section 5.1). `MultiAgentSignalEnv` inherently drives signals via the
shield on every step, which would make it the wrong tool here even in a read-only usage
pattern -- `SumoTwinEnv.step()` just advances the simulation under whatever default/agent
control is already running.

Known integration gap (same class as IntersectionAgentService's, see that module's
docstring): with no Redis/state-writer pipeline yet, this coordinator's "live state"
comes from its own separate SUMO process, not literally the same simulation an
IntersectionAgentService for the same junctions would be running. A real deployment needs
all of these processes reading one shared twin's state (via the bus + Redis latest-value
store) -- closing that gap is state-writer/Redis work, not something one service can
correctly paper over on its own; documented here rather than silently assumed done.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from atlas.bus.producer import BusProducer
from atlas.common.logging import get_logger
from atlas.coordination.corridor_geometry import compute_corridor_segment_distances_m
from atlas.coordination.greenwave import compute_greenwave_offsets, detect_spillback
from atlas.twin.env import SumoTwinEnv, TwinEnvConfig

logger = get_logger(__name__)


class CorridorCoordinatorConfig(BaseModel):
    corridor_id: str
    tls_ids: list[str] = Field(min_length=2)  # ordered along the arterial
    net_file: Path
    route_file: Path
    additional_files: list[Path] = Field(default_factory=list)
    recompute_interval_s: float = 60.0
    spillback_threshold: float = 0.85
    min_platoon_speed_mps: float = 2.0  # floor, avoids huge/undefined offsets near-gridlock
    max_offset_for_hint_normalization_s: float = 60.0
    seed: int = 42


class CorridorCoordinatorService:
    def __init__(
        self, config: CorridorCoordinatorConfig, bus_producer: BusProducer | None = None
    ) -> None:
        self.config = config
        self._bus = bus_producer
        self._segment_distances_m = compute_corridor_segment_distances_m(
            config.net_file, config.tls_ids
        )
        self._twin = SumoTwinEnv(
            TwinEnvConfig(
                name=f"corridor_coordinator.{config.corridor_id}",
                net_file=config.net_file,
                route_file=config.route_file,
                additional_files=config.additional_files,
            )
        )
        self._last_member_hints: list[dict[str, Any]] = []
        self._last_alerts: list[dict[str, Any]] = []
        self._running = False

    def start(self) -> None:
        self._twin.reset(seed=self.config.seed)
        self._running = True

    def stop(self) -> None:
        self._running = False
        self._twin.close()

    def recompute(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Advances the simulation by one recompute window, measures platoon speed and
        exit-link occupancy across every member, and publishes fresh hints/alerts.
        Returns (member_hints, alerts) for callers/tests to inspect directly.
        """
        steps = max(1, round(self.config.recompute_interval_s / self._twin.config.step_length_s))
        self._twin.step(n=steps)

        speeds_mps: list[float] = []
        occupancies: dict[str, float] = {}
        for tls_id in self.config.tls_ids:
            state = self._twin.get_junction_state(tls_id)
            for lane in state.incoming_lanes:
                if lane.vehicle_count > 0:
                    speeds_mps.append(lane.mean_speed_mps)
                occupancies[tls_id] = max(occupancies.get(tls_id, 0.0), lane.occupancy)

        platoon_speed_mps = (
            max(sum(speeds_mps) / len(speeds_mps), self.config.min_platoon_speed_mps)
            if speeds_mps
            else self.config.min_platoon_speed_mps
        )

        offsets_s = compute_greenwave_offsets(self._segment_distances_m, platoon_speed_mps)
        member_hints = [
            {
                "intersection_id": tls_id,
                "offset_s": offset_s,
                "hint_value": min(offset_s / self.config.max_offset_for_hint_normalization_s, 1.0),
            }
            for tls_id, offset_s in zip(self.config.tls_ids, offsets_s, strict=True)
        ]

        alerts = [
            {"intersection_id": tls_id, "occupancy": occupancy}
            for tls_id, occupancy in occupancies.items()
            if detect_spillback(occupancy, self.config.spillback_threshold)
        ]

        self._last_member_hints = member_hints
        self._last_alerts = alerts
        self._publish(member_hints, alerts)
        return member_hints, alerts

    def _publish(self, member_hints: list[dict[str, Any]], alerts: list[dict[str, Any]]) -> None:
        if self._bus is None:
            return
        self._bus.publish(
            "corridor.hints.v1",
            {"corridor_id": self.config.corridor_id, "member_hints": member_hints},
        )
        for alert in alerts:
            self._bus.publish(
                "corridor.alerts.v1",
                {
                    "corridor_id": self.config.corridor_id,
                    "intersection_id": alert["intersection_id"],
                    "alert_type": "spillback",
                    "occupancy": alert["occupancy"],
                },
            )

    def state(self) -> dict[str, Any]:
        return {
            "corridor_id": self.config.corridor_id,
            "member_hints": self._last_member_hints,
            "alerts": self._last_alerts,
        }


def create_app(coordinators: dict[str, CorridorCoordinatorService]) -> FastAPI:
    app = FastAPI(title="Corridor Coordinator")

    @app.get("/health")
    def get_health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/corridor/{corridor_id}/state")
    def get_corridor_state(corridor_id: str) -> dict[str, Any]:
        coordinator = coordinators.get(corridor_id)
        if coordinator is None:
            raise HTTPException(status_code=404, detail=f"Unknown corridor '{corridor_id}'")
        return coordinator.state()

    return app


def main() -> None:
    """CLI entrypoint: `python -m atlas.services.corridor_coordinator --corridor-id ...
    --tls-ids A,B,C [--kafka-brokers localhost:9092]`. Used directly and by docker-
    compose's corridor_coordinator service. Runs one recompute cycle every
    `recompute_interval_s`, forever.
    """
    import argparse
    import time

    import uvicorn

    from atlas.common.config import project_root

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--corridor-id", required=True)
    parser.add_argument(
        "--tls-ids", required=True, help="comma-separated, ordered along the arterial"
    )
    parser.add_argument("--network", default="manhattan_midtown")
    parser.add_argument("--route-file", default="demand/pm_peak.rou.xml")
    parser.add_argument("--recompute-interval-s", type=float, default=60.0)
    parser.add_argument("--kafka-brokers", default=None)
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()

    network_dir = project_root() / "data" / "networks" / args.network
    vtypes_file = network_dir / "vtypes.add.xml"
    config = CorridorCoordinatorConfig(
        corridor_id=args.corridor_id,
        tls_ids=args.tls_ids.split(","),
        net_file=network_dir / "net.net.xml",
        route_file=network_dir / args.route_file,
        additional_files=[vtypes_file] if vtypes_file.is_file() else [],
        recompute_interval_s=args.recompute_interval_s,
    )
    bus_producer = BusProducer.connect([args.kafka_brokers]) if args.kafka_brokers else None
    service = CorridorCoordinatorService(config, bus_producer=bus_producer)
    service.start()

    import threading

    def recompute_loop() -> None:
        while True:
            service.recompute()
            time.sleep(config.recompute_interval_s)

    threading.Thread(target=recompute_loop, daemon=True).start()
    uvicorn.run(create_app({args.corridor_id: service}), host="0.0.0.0", port=args.port)  # noqa: S104


if __name__ == "__main__":
    main()
