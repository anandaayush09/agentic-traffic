"""ControlledTwin: the real Manhattan twin with ATLAS's own signal controllers closing the
loop, plus the hooks to put disruptions and playbook responses into it.

This is the one closed-loop control core shared by the live-traffic service (dashboard
map, `backend="traci"`) and the what-if tool (`backend="libsumo"`), so both exercise the
exact same thing: `MultiAgentSignalEnv` (which routes every decision through the safety
shield), per-junction controller modes from `atlas.services.decision`, the scenario
engine for disruptions, and `playbook_effects` for responses. It mirrors the decision
loop the evaluation harness uses (`atlas.evaluation.runner.run_scenario_cell`).

Only `tls_ids` (default: the 5 corridor junctions the evaluation and training use) are
ATLAS-controlled; every other signal keeps SUMO's native program.

Single-threaded by design: every method that touches the simulation must be called from
the one thread that owns the twin. `set_mode` is the exception (plain Python state).
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, get_args

from atlas.common.logging import get_logger
from atlas.control.actuated import ActuatedConfig
from atlas.control.fixed_time import native_green_durations_from_net_file
from atlas.control.max_pressure import MaxPressureProgram
from atlas.control.shield import ShieldConfig
from atlas.rl import MultiAgentSignalEnv, MultiAgentSignalEnvConfig
from atlas.services.decision import AgentMode, select_action_for_mode
from atlas.twin.env import TwinBackend, TwinEnvConfig
from atlas.twin.incident_effects import RoadIndex
from atlas.twin.playbook_effects import PlaybookEffect
from atlas.twin.scenario import ScenarioController, ScenarioSpec

logger = get_logger(__name__)

CORRIDOR_TLS_IDS: tuple[str, ...] = ("42430333", "42430338", "42430342", "42430344", "42430347")

# The modes a junction can run live. `rl` is deliberately absent: no trained checkpoint
# exists (README, "What isn't done"), and `select_action_for_mode` fails loudly for it.
CONTROLLER_MODES: tuple[str, ...] = ("fixed_time", "actuated", "max_pressure")
assert set(CONTROLLER_MODES) <= set(get_args(AgentMode))

# A travel time so large the router treats the road as not worth taking.
_AVOID_TRAVEL_TIME_S = 5000.0


@dataclass(frozen=True, slots=True)
class ActiveEffect:
    effect_id: str
    effect: PlaybookEffect
    started_at_s: float
    ends_at_s: float


class ControlledTwin:
    def __init__(
        self,
        net_file: Path,
        route_file: Path,
        additional_files: Sequence[Path] = (),
        *,
        tls_ids: Sequence[str] = CORRIDOR_TLS_IDS,
        backend: TwinBackend = "libsumo",
        decision_interval_s: float = 5.0,
        default_mode: str = "fixed_time",
        road_index: RoadIndex | None = None,
    ) -> None:
        if default_mode not in CONTROLLER_MODES:
            raise ValueError(f"default_mode must be one of {CONTROLLER_MODES}, got {default_mode}")
        self._net_file = net_file
        self._route_file = route_file
        self._additional_files = list(additional_files)
        self._tls_ids = list(tls_ids)
        self._backend: TwinBackend = backend
        self._decision_interval_s = decision_interval_s
        self._default_mode = default_mode
        self._index = road_index or RoadIndex.from_net_file(net_file)

        self._programs = {t: MaxPressureProgram.from_net_file(t, net_file) for t in self._tls_ids}
        self._native_durations = {
            t: native_green_durations_from_net_file(t, net_file) for t in self._tls_ids
        }
        self._pressure_lanes = {
            t: sorted({m.from_lane for m in p.movements} | {m.to_lane for m in p.movements})
            for t, p in self._programs.items()
        }
        self._incoming_lanes = {
            t: sorted({m.from_lane for m in p.movements}) for t, p in self._programs.items()
        }

        self._env: MultiAgentSignalEnv | None = None
        self._scenarios: ScenarioController | None = None
        self._base_modes: dict[str, str] = {}
        self._effects: dict[str, ActiveEffect] = {}
        self._avoid_counts: dict[str, int] = {}
        self._last_actions: dict[str, int] = {}
        self._elapsed_in_phase_s: dict[str, float] = {}

    # -- lifecycle --

    def reset(self, seed: int = 42) -> None:
        self.close()
        env = MultiAgentSignalEnv(
            MultiAgentSignalEnvConfig(
                twin_config=TwinEnvConfig(
                    name="controlled_twin",
                    net_file=self._net_file,
                    route_file=self._route_file,
                    additional_files=self._additional_files,
                    backend=self._backend,
                    # A closure/penalty invalidates routes through a road; without
                    # rerouting those vehicles would be dropped instead of going round.
                    enable_rerouting=True,
                ),
                tls_ids=self._tls_ids,
                max_lanes=10,
                max_green_phases=2,
                shield_config=ShieldConfig(min_green_s=10.0, max_green_s=60.0),
                decision_interval_s=self._decision_interval_s,
            )
        )
        env.reset(seed=seed)
        self._env = env
        # ScenarioController's type hint says SumoTwinEnv, but `.sync()` only needs
        # `.connection`/`.sim_time` (atlas.twin.env.TwinLike) -- same usage as the
        # evaluation runner.
        self._scenarios = ScenarioController(env, [])  # type: ignore[arg-type]
        self._base_modes = dict.fromkeys(self._tls_ids, self._default_mode)
        self._effects = {}
        self._avoid_counts = {}
        self._last_actions = dict.fromkeys(self._tls_ids, 0)
        self._elapsed_in_phase_s = dict.fromkeys(self._tls_ids, 0.0)

    def close(self) -> None:
        env, self._env, self._scenarios = self._env, None, None
        if env is not None:
            env.close()

    @property
    def is_running(self) -> bool:
        return self._env is not None

    @property
    def connection(self) -> Any:
        return self._require_env().connection

    @property
    def sim_time(self) -> float:
        return self._require_env().sim_time

    @property
    def tls_ids(self) -> list[str]:
        return list(self._tls_ids)

    def finished(self) -> bool:
        """True once every scheduled vehicle has arrived (the demand file is spent)."""
        return int(self.connection.simulation.getMinExpectedNumber()) == 0

    # -- control --

    def set_mode(self, tls_id: str, mode: str) -> None:
        """Sets a junction's base controller mode. Plain state, safe from any thread; it
        takes effect on the next decision. A playbook effect that overrides the junction
        wins until it expires."""
        if tls_id not in self._tls_ids:
            raise KeyError(f"'{tls_id}' is not an ATLAS-controlled junction")
        if mode not in CONTROLLER_MODES:
            raise ValueError(f"mode must be one of {CONTROLLER_MODES}, got '{mode}'")
        self._base_modes[tls_id] = mode

    def base_modes(self) -> dict[str, str]:
        return dict(self._base_modes)

    def effective_modes(self) -> dict[str, str]:
        modes = dict(self._base_modes)
        for active in self._effects.values():
            modes.update(active.effect.junction_modes)
        return modes

    def overridden_junctions(self) -> set[str]:
        return {j for active in self._effects.values() for j in active.effect.junction_modes}

    def active_effects(self) -> list[ActiveEffect]:
        return list(self._effects.values())

    def add_disruptions(self, specs: Sequence[ScenarioSpec]) -> None:
        scenarios = self._require_scenarios()
        for spec in specs:
            scenarios.add_spec(spec)

    def apply_effect(self, effect_id: str, effect: PlaybookEffect) -> ActiveEffect:
        """Puts a compiled playbook response into the running twin until it expires."""
        if effect_id in self._effects:
            raise ValueError(f"Effect '{effect_id}' is already active")
        now = self.sim_time
        active = ActiveEffect(effect_id, effect, now, now + effect.duration_s)
        for edge_id in effect.avoid_edges:
            if self._avoid_counts.get(edge_id, 0) == 0:
                self.connection.edge.adaptTraveltime(edge_id, _AVOID_TRAVEL_TIME_S)
            self._avoid_counts[edge_id] = self._avoid_counts.get(edge_id, 0) + 1
        self._effects[effect_id] = active
        logger.info(
            "controlled_twin.effect_applied", effect_id=effect_id, name=effect.playbook_name
        )
        return active

    def step_decisions(self, n: int = 1) -> None:
        """Runs `n` decision cycles (each `decision_interval_s` simulated seconds): the
        scenario engine syncs, expired effects revert, then every controlled junction's
        mode picks an action which goes through the safety shield."""
        env = self._require_env()
        scenarios = self._require_scenarios()
        for _ in range(n):
            scenarios.sync()
            self._expire_effects()
            self._reroute_around_avoided_roads()
            modes = self.effective_modes()
            conn = env.connection
            actions: dict[str, int] = {}
            for tls_id in self._tls_ids:
                mode: AgentMode = modes[tls_id]  # type: ignore[assignment]
                queue_lengths = {
                    lane: int(conn.lane.getLastStepHaltingNumber(lane))
                    for lane in self._pressure_lanes[tls_id]
                }
                action = select_action_for_mode(
                    mode,
                    self._programs[tls_id],
                    queue_lengths=queue_lengths,
                    native_durations=self._native_durations[tls_id],
                    sim_time=env.sim_time if mode == "fixed_time" else None,
                    last_action=self._last_actions[tls_id],
                    elapsed_in_phase_s=(
                        self._elapsed_in_phase_s[tls_id] if mode == "actuated" else None
                    ),
                    actuated_config=ActuatedConfig(),
                )
                self._elapsed_in_phase_s[tls_id] = (
                    self._elapsed_in_phase_s[tls_id] + self._decision_interval_s
                    if action == self._last_actions[tls_id]
                    else self._decision_interval_s
                )
                actions[tls_id] = action
            self._last_actions = actions
            env.step(actions)

    def junction_states(self) -> dict[str, dict[str, object]]:
        """Per controlled junction: effective mode, whether a playbook is overriding it,
        current signal phase and total queued (halted) vehicles on its approaches."""
        conn = self.connection
        modes = self.effective_modes()
        overridden = self.overridden_junctions()
        return {
            tls_id: {
                "mode": modes[tls_id],
                "playbook_override": tls_id in overridden,
                "phase": int(conn.trafficlight.getPhase(tls_id)),
                "queue": sum(
                    int(conn.lane.getLastStepHaltingNumber(lane))
                    for lane in self._incoming_lanes[tls_id]
                ),
            }
            for tls_id in self._tls_ids
        }

    # -- internals --

    def _expire_effects(self) -> None:
        now = self.sim_time
        for effect_id in [i for i, a in self._effects.items() if a.ends_at_s <= now]:
            active = self._effects.pop(effect_id)
            for edge_id in active.effect.avoid_edges:
                self._avoid_counts[edge_id] -= 1
                if self._avoid_counts[edge_id] == 0:
                    # Restore the free-flow travel time so routing treats the road
                    # normally again.
                    free_flow = self._index.length_m[edge_id] / self._index.free_flow_mps[edge_id]
                    self.connection.edge.adaptTraveltime(edge_id, free_flow)
            logger.info("controlled_twin.effect_expired", effect_id=effect_id)

    def _reroute_around_avoided_roads(self) -> None:
        """Re-plans every vehicle whose remaining route crosses a penalised road. The
        built-in rerouting device ignores `adaptTraveltime`, so this does it explicitly
        (verified: it measurably empties the penalised road)."""
        avoided = {e for e, count in self._avoid_counts.items() if count > 0}
        if not avoided:
            return
        conn = self.connection
        for vehicle_id in conn.vehicle.getIDList():
            route = conn.vehicle.getRoute(vehicle_id)
            ahead = route[conn.vehicle.getRouteIndex(vehicle_id) :]
            if any(edge in avoided for edge in ahead):
                conn.vehicle.rerouteTraveltime(vehicle_id, False)

    def _require_env(self) -> MultiAgentSignalEnv:
        if self._env is None:
            raise RuntimeError("ControlledTwin is not running. Call reset() first.")
        return self._env

    def _require_scenarios(self) -> ScenarioController:
        if self._scenarios is None:
            raise RuntimeError("ControlledTwin is not running. Call reset() first.")
        return self._scenarios


def edge_halted_total(conn: Any, edge_ids: Collection[str]) -> int:
    """Halted vehicles summed over `edge_ids` (a helper for impact/what-if metrics)."""
    return sum(int(conn.edge.getLastStepHaltingNumber(e)) for e in edge_ids)
