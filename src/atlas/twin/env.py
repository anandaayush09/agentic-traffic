"""SumoTwinEnv: a thin, typed wrapper around one SUMO simulation instance.

One instance = one SUMO process/binding = one simulated district. This is the building
block both the MARL training rollout workers (many parallel in-process instances, see
Part 6.1 of the design docs) and the live/demo twin service (one instance behind the
event bus, Part 3.8) will be built from in later weeks. This class deliberately knows
nothing about Ray, Kafka, or RL -- it only starts/steps/reads/closes a SUMO simulation.

Backend: the frozen architecture specifies libsumo (in-process bindings, no TraCI socket
overhead) as the default and only backend used by training and the live twin. A
`backend="traci"` escape hatch exists because libsumo cannot drive `sumo-gui` or serve
multiple simultaneous client connections -- needed later for interactive debugging and
demos, never for training or evaluation. This is an implementation detail, not a change
to the frozen decision: the default remains libsumo everywhere it matters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, cast

from pydantic import BaseModel, Field

from atlas.common.logging import get_logger
from atlas.common.sumo_locate import get_binary_path, get_sumo_home

logger = get_logger(__name__)

TwinBackend = Literal["libsumo", "traci"]


class TwinEnvConfig(BaseModel):
    """Configuration for one SumoTwinEnv instance."""

    name: str
    net_file: Path
    route_file: Path
    additional_files: list[Path] = Field(default_factory=list)
    traffic_light_ids: list[str] = Field(default_factory=list)
    step_length_s: float = Field(default=1.0, gt=0)
    sumo_seed: int = 42
    backend: TwinBackend = "libsumo"
    gui: bool = False


@dataclass(frozen=True, slots=True)
class LaneState:
    """Raw per-lane traffic state for one incoming approach lane at one instant."""

    lane_id: str
    queue_length: int
    """Number of halting (near-stationary) vehicles on the lane -- the max-pressure /
    PressLight-style queue signal used by the control-plane reward (Part 6.1)."""
    occupancy: float
    """Fraction (0..1) of the lane's length covered by vehicles over the last step."""
    mean_speed_mps: float
    vehicle_count: int


@dataclass(frozen=True, slots=True)
class JunctionState:
    """Raw signal + traffic state for one signalized junction at one instant."""

    tls_id: str
    phase_index: int
    phase_state: str
    """Per-signal-group state string, e.g. "rrrGGgGrr" (SUMO's r/y/g/G/etc. encoding)."""
    incoming_lanes: tuple[LaneState, ...]


class TwinEnvError(RuntimeError):
    """Raised for twin lifecycle errors: not running, already running, backend unavailable."""


class SumoTwinEnv:
    """Controls one SUMO simulation via libsumo (default) or traci."""

    def __init__(self, config: TwinEnvConfig) -> None:
        self._config = config
        self._conn: Any = None
        self._running = False
        get_sumo_home()  # resolve SUMO_HOME as a side effect; fail fast if unavailable

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def config(self) -> TwinEnvConfig:
        return self._config

    @property
    def sim_time(self) -> float:
        self._require_running()
        return cast(float, self._conn.simulation.getTime())

    def reset(self, seed: int | None = None) -> None:
        """Start (or restart) the simulation from t=0."""
        if self._running:
            self.close()

        conn = self._load_backend()
        binary_name = "sumo-gui" if self._config.gui else "sumo"
        cmd = [
            str(get_binary_path(binary_name)),
            "--net-file",
            str(self._config.net_file),
            "--route-files",
            str(self._config.route_file),
            "--step-length",
            str(self._config.step_length_s),
            "--seed",
            str(seed if seed is not None else self._config.sumo_seed),
            "--no-step-log",
            "true",
            "--no-warnings",
            "true",
            "--duration-log.disable",
            "true",
            "--time-to-teleport",
            "-1",
        ]
        for extra in self._config.additional_files:
            cmd += ["--additional-files", str(extra)]

        logger.info(
            "twin.starting",
            twin_name=self._config.name,
            backend=self._config.backend,
            net_file=str(self._config.net_file),
        )
        conn.start(cmd)
        self._conn = conn
        self._running = True
        logger.info("twin.started", twin_name=self._config.name, sim_time=self.sim_time)

    def step(self, n: int = 1) -> None:
        """Advance the simulation by `n` steps of `step_length_s` seconds each."""
        self._require_running()
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        for _ in range(n):
            self._conn.simulationStep()

    def get_traffic_light_ids(self) -> list[str]:
        self._require_running()
        return cast(list[str], list(self._conn.trafficlight.getIDList()))

    def get_junction_state(self, tls_id: str) -> JunctionState:
        """Read the current signal phase and per-incoming-lane traffic state for one
        signalized junction."""
        self._require_running()
        conn = self._conn
        phase_index = cast(int, conn.trafficlight.getPhase(tls_id))
        phase_state = cast(str, conn.trafficlight.getRedYellowGreenState(tls_id))
        lanes = tuple(self._read_lane_state(lane_id) for lane_id in self._incoming_lanes(tls_id))
        return JunctionState(
            tls_id=tls_id,
            phase_index=phase_index,
            phase_state=phase_state,
            incoming_lanes=lanes,
        )

    def get_all_junction_states(self) -> dict[str, JunctionState]:
        """Read state for every configured traffic light (or every one SUMO reports, if
        none were configured)."""
        ids = self._config.traffic_light_ids or self.get_traffic_light_ids()
        return {tls_id: self.get_junction_state(tls_id) for tls_id in ids}

    def close(self) -> None:
        if self._running and self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._running = False
                self._conn = None
                logger.info("twin.closed", twin_name=self._config.name)

    def __enter__(self) -> SumoTwinEnv:
        self.reset()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # -- internal helpers --

    def _require_running(self) -> None:
        if not self._running or self._conn is None:
            raise TwinEnvError("Environment is not running. Call reset() first.")

    def _incoming_lanes(self, tls_id: str) -> list[str]:
        controlled = cast(list[str], self._conn.trafficlight.getControlledLanes(tls_id))
        seen: dict[str, None] = {}
        for lane_id in controlled:
            seen.setdefault(lane_id, None)
        return list(seen.keys())

    def _read_lane_state(self, lane_id: str) -> LaneState:
        conn = self._conn
        return LaneState(
            lane_id=lane_id,
            queue_length=cast(int, conn.lane.getLastStepHaltingNumber(lane_id)),
            occupancy=cast(float, conn.lane.getLastStepOccupancy(lane_id)) / 100.0,
            mean_speed_mps=cast(float, conn.lane.getLastStepMeanSpeed(lane_id)),
            vehicle_count=cast(int, conn.lane.getLastStepVehicleNumber(lane_id)),
        )

    def _load_backend(self) -> Any:
        if self._config.backend == "libsumo":
            try:
                import libsumo

                return libsumo
            except ImportError as exc:
                raise TwinEnvError(
                    "backend='libsumo' was requested but the libsumo package is not "
                    "importable. Install it (see pyproject.toml) or set backend='traci'."
                ) from exc
        else:
            import traci

            return traci
