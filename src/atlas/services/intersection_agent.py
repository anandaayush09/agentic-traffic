"""Intersection Agent: the Week 6 roadmap's "Intersection Agent... as real FastAPI/async
services on the bus" (docs/design/13_roadmap.md, Week 6; docs/design/
04_agent_architecture.md section 4.1). Owns ONE junction's decision loop: every
`decision_interval_s`, pick an action (RL policy, max-pressure, fixed-time, or hold, per
`mode`), step the junction via `MultiAgentSignalEnv` (constructed with a single-element
`tls_ids`, a valid decentralized-execution view of the same class used for training), and
publish `control.commands.v1` + a `system.health.v1` heartbeat to the bus.

Why `MultiAgentSignalEnv` and not `SingleJunctionEnv` (Week 3): a loaded MAPPO checkpoint
(scripts/train_mappo_corridor.py / train_mappo_district.py) expects the exact observation
shape `MultiAgentSignalEnv` produces (padded lanes + lane_valid + phase one-hot +
elapsed), which differs from `SingleJunctionEnv`'s. Using the SAME env class for
execution as for training is also the literal CTDE story (docs/design/06_ai_pipeline.md
section 6.1): "centralized critic... during training only" -- at execution, the actor
only ever needs one agent's local observation, which `MultiAgentSignalEnv` with
`num_agents=1` gives directly, with no code duplication risk of an independently
maintained observation-builder silently drifting from what the policy was trained on.

Mode switching (`decision.py`'s `select_action_for_mode`) needs raw queue lengths and
sim time for the non-RL modes, which `MultiAgentSignalEnv` didn't expose before this
service needed it -- see its `.connection`/`.sim_time` properties, added for this,
mirroring `SumoTwinEnv`/`SingleJunctionEnv`'s existing escape-hatch precedent.

JWT auth on `/mode` (docs/design/13_roadmap.md, Week 10: "JWT auth on mutating APIs";
docs/design/03_system_architecture.md section 3.6's "Control API...JWT, operator-only")
-- deferred through Week 6-9, closed here using the same `atlas.services.auth.
require_operator` dependency the Week 10 Cognitive API already uses, not a
second auth scheme.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from atlas.bus.producer import BusProducer
from atlas.common.logging import get_logger
from atlas.control.fixed_time import native_green_durations_from_net_file
from atlas.control.max_pressure import MaxPressureProgram
from atlas.control.shield import ShieldConfig
from atlas.marl.network import ActorCritic
from atlas.rl.multi_agent_env import MultiAgentSignalEnv, MultiAgentSignalEnvConfig
from atlas.services.auth import require_operator
from atlas.services.decision import AgentMode, actor_name_for_mode, select_action_for_mode
from atlas.twin.env import TwinEnvConfig

logger = get_logger(__name__)


class IntersectionAgentConfig(BaseModel):
    tls_id: str
    net_file: Path
    route_file: Path
    additional_files: list[Path] = Field(default_factory=list)
    max_lanes: int = Field(gt=0)
    max_green_phases: int = Field(gt=0)
    decision_interval_s: float = 5.0
    shield_config: ShieldConfig = Field(default_factory=ShieldConfig)
    default_mode: AgentMode = "max_pressure"
    model_path: Path | None = None
    seed: int = 42


class IntersectionAgentService:
    def __init__(
        self, config: IntersectionAgentConfig, bus_producer: BusProducer | None = None
    ) -> None:
        self.config = config
        self.mode: AgentMode = config.default_mode
        self._bus = bus_producer
        self._service_id = f"intersection_agent.{config.tls_id}"

        self._program = MaxPressureProgram.from_net_file(config.tls_id, config.net_file)
        self._native_durations = native_green_durations_from_net_file(
            config.tls_id, config.net_file
        )

        self._env = MultiAgentSignalEnv(
            MultiAgentSignalEnvConfig(
                twin_config=TwinEnvConfig(
                    name=self._service_id,
                    net_file=config.net_file,
                    route_file=config.route_file,
                    additional_files=config.additional_files,
                ),
                tls_ids=[config.tls_id],
                max_lanes=config.max_lanes,
                max_green_phases=config.max_green_phases,
                shield_config=config.shield_config,
                decision_interval_s=config.decision_interval_s,
            )
        )

        self._net: ActorCritic | None = None
        if config.model_path is not None:
            # Reads obs_dim from the env itself (not re-derived here) so this can never
            # silently drift out of sync with MultiAgentSignalEnv's own observation
            # shape (e.g. Week 7's corridor-hint feature added a dimension after this
            # service was first written).
            self._net = ActorCritic(
                obs_dim=self._env.observation_dim,
                action_dim=config.max_green_phases,
                num_agents=1,
            )
            self._net.load_state_dict(torch.load(config.model_path, weights_only=True))
            self._net.eval()

        self._obs: np.ndarray | None = None
        self._last_action = 0
        self._last_heartbeat: datetime | None = None
        self._running = False

    def start(self) -> None:
        observations = self._env.reset(seed=self.config.seed)
        self._obs = observations[self.config.tls_id]
        self._running = True

    def stop(self) -> None:
        self._running = False
        self._env.close()

    def decide_and_step(self) -> dict[str, Any]:
        """Runs one decision cycle: pick an action for the current mode, step the
        env, publish the resulting command + heartbeat. Returns the step's info dict
        for callers/tests that want to inspect what happened.
        """
        queue_lengths = None
        if self.mode == "max_pressure":
            conn = self._env.connection
            queue_lengths = {
                lane_id: int(conn.lane.getLastStepHaltingNumber(lane_id))
                for lane_id in {m.from_lane for m in self._program.movements}
                | {m.to_lane for m in self._program.movements}
            }

        obs_tensor = torch.tensor(self._obs, dtype=torch.float32) if self._obs is not None else None
        action = select_action_for_mode(
            self.mode,
            self._program,
            queue_lengths=queue_lengths,
            native_durations=self._native_durations,
            sim_time=self._env.sim_time if self.mode == "fixed_time" else None,
            net=self._net,
            obs=obs_tensor,
            last_action=self._last_action,
        )
        self._last_action = action

        observations, _, _, _, infos = self._env.step({self.config.tls_id: action})
        self._obs = observations[self.config.tls_id]
        info = infos[self.config.tls_id]

        phase_id = self._program.green_phase_indices[
            action % len(self._program.green_phase_indices)
        ]
        self._publish_command(phase_id, info)
        self._publish_heartbeat(info)
        return info

    def _publish_command(self, phase_id: int, info: dict[str, Any]) -> None:
        if self._bus is None:
            return
        self._bus.publish(
            "control.commands.v1",
            {
                "intersection_id": self.config.tls_id,
                "actor": actor_name_for_mode(self.mode),
                "action": {"kind": "set_phase", "phase_id": phase_id},
                "shield": {
                    "modified": bool(info["shield_modified"]),
                    "reason": info["shield_reason"],
                },
                "policy_version": "mappo_v1" if self.mode == "rl" else "n/a",
            },
        )

    def _publish_heartbeat(self, info: dict[str, Any]) -> None:
        self._last_heartbeat = datetime.now(UTC)
        if self._bus is None:
            return
        self._bus.publish(
            "system.health.v1",
            {
                "service_id": self._service_id,
                "status": "ok",
                "detail": {"mode": self.mode, "pressure": info["pressure"]},
            },
        )

    async def run_forever(self) -> None:
        self.start()
        try:
            while self._running:
                self.decide_and_step()
                await asyncio.sleep(self.config.decision_interval_s)
        finally:
            self.stop()

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok" if self._running else "stopped",
            "tls_id": self.config.tls_id,
            "mode": self.mode,
            "last_heartbeat_ts": (
                self._last_heartbeat.isoformat() if self._last_heartbeat is not None else None
            ),
        }


class ModeUpdate(BaseModel):
    mode: AgentMode


def create_app(
    service: IntersectionAgentService, jwt_secret: str, start_background_loop: bool = True
) -> FastAPI:
    """`start_background_loop=False` skips auto-starting `run_forever()` on app
    startup -- used by tests that want to drive `service.start()`/`decide_and_step()`
    themselves rather than racing an async background task. `jwt_secret` gates the
    mutating `POST /mode` endpoint (an operator forcing rl/max_pressure/fixed_time/off
    at a live junction is exactly the "Control API...JWT, operator-only" mutating
    action the design doc names) -- required, not optional, so a caller can't
    accidentally deploy this service unauthenticated.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        task = asyncio.create_task(service.run_forever()) if start_background_loop else None
        try:
            yield
        finally:
            service.stop()
            if task is not None:
                task.cancel()

    app = FastAPI(title=f"Intersection Agent ({service.config.tls_id})", lifespan=lifespan)
    require_jwt = require_operator(jwt_secret)

    @app.get("/health")
    def get_health() -> dict[str, Any]:
        return service.health()

    @app.get("/mode")
    def get_mode() -> dict[str, str]:
        return {"mode": service.mode}

    @app.post("/mode")
    def set_mode(
        update: ModeUpdate,
        _claims: dict[str, object] = Depends(require_jwt),  # noqa: B008
    ) -> dict[str, str]:
        service.mode = update.mode
        return {"mode": service.mode}

    return app


def main() -> None:
    """CLI entrypoint: `python -m atlas.services.intersection_agent --tls-id ... --network
    manhattan_midtown [--mode max_pressure] [--kafka-brokers localhost:9092]`. Used
    directly and by docker-compose's intersection_agent service.
    """
    import argparse
    import os

    import uvicorn

    from atlas.common.config import project_root

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--tls-id", required=True)
    parser.add_argument("--network", default="manhattan_midtown")
    parser.add_argument("--route-file", default="demand/pm_peak.rou.xml")
    parser.add_argument("--max-lanes", type=int, default=10)
    parser.add_argument("--max-green-phases", type=int, default=2)
    parser.add_argument("--decision-interval-s", type=float, default=5.0)
    parser.add_argument(
        "--mode", default="max_pressure", choices=["rl", "max_pressure", "fixed_time", "off"]
    )
    parser.add_argument("--model-path", default=None)
    parser.add_argument(
        "--kafka-brokers", default=None, help="e.g. localhost:9092; omit to run bus-less"
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--jwt-secret",
        default=None,
        help="Signs/verifies operator JWTs for POST /mode. Falls back to the "
        "ATLAS_JWT_SECRET env var; refuses to start with neither set.",
    )
    args = parser.parse_args()

    jwt_secret = args.jwt_secret or os.environ.get("ATLAS_JWT_SECRET")
    if not jwt_secret:
        parser.error("--jwt-secret or the ATLAS_JWT_SECRET env var is required")

    network_dir = project_root() / "data" / "networks" / args.network
    vtypes_file = network_dir / "vtypes.add.xml"
    config = IntersectionAgentConfig(
        tls_id=args.tls_id,
        net_file=network_dir / "net.net.xml",
        route_file=network_dir / args.route_file,
        additional_files=[vtypes_file] if vtypes_file.is_file() else [],
        max_lanes=args.max_lanes,
        max_green_phases=args.max_green_phases,
        decision_interval_s=args.decision_interval_s,
        default_mode=args.mode,
        model_path=Path(args.model_path) if args.model_path else None,
    )
    bus_producer = BusProducer.connect([args.kafka_brokers]) if args.kafka_brokers else None
    service = IntersectionAgentService(config, bus_producer=bus_producer)
    uvicorn.run(create_app(service, jwt_secret), host="0.0.0.0", port=args.port)  # noqa: S104


if __name__ == "__main__":
    main()
