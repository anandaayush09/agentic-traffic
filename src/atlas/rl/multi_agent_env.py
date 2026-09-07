"""MultiAgentSignalEnv: one shared digital twin, N agents (one per signalized junction),
stepping simultaneously -- the direct multi-agent generalization of SingleJunctionEnv
(Week 3), and the Week 4 roadmap deliverable ahead of the from-scratch MAPPO
implementation in Week 5 (docs/design/13_roadmap.md, Week 4; docs/design/06_ai_pipeline.md
section 6.1).

API: a dict-in/dict-out parallel multi-agent interface (`reset() -> {agent_id: obs}`,
`step(actions: {agent_id: action}) -> (obs, rewards, terminations, truncations, infos)`,
all dicts keyed by `tls_id`) -- the standard convention popularized by PettingZoo's
ParallelEnv, but NOT an actual PettingZoo dependency: PettingZoo's class hierarchy exists
mainly to interoperate with RLlib, and the frozen architecture explicitly writes MAPPO
from scratch rather than using RLlib (docs/design/07_tech_stack.md section 7.2), so
depending on PettingZoo wouldn't earn its keep here.

Parameter sharing (Week 5's actual point of building this) requires every agent to expose
the IDENTICAL observation/action shape, even though junctions have different numbers of
incoming lanes -- verified against both networks before choosing constants: every
junction in the sandbox network and the real Manhattan district has exactly 2 green
phases, but incoming lane counts range from 2 to 10. So:

- `max_lanes`/`max_green_phases` are explicit, validated config (not hardcoded) -- the
  same code serves the small sandbox network (fast tests) and the real district (the
  actual training/throughput target).
- Padded lane slots carry a `lane_valid` flag (a 4th per-lane feature, on top of Single
  JunctionEnv's [queue_norm, occupancy, speed_norm]) so the policy can distinguish a real
  empty lane from padding.
- An action beyond a junction's actual green-phase count is taken modulo that junction's
  count -- deterministic, never crashes, and moot for both networks today (uniformly 2
  phases) but keeps the code correct for a future network that isn't.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, Field, model_validator

from atlas.control.max_pressure import MaxPressureProgram, total_pressure
from atlas.control.phases import yellow_phase_after
from atlas.control.shield import ShieldConfig, SignalProgram, SignalShield
from atlas.twin import SumoTwinEnv, TwinEnvConfig

_FEATURES_PER_LANE = 4  # queue_norm, occupancy, speed_norm, lane_valid


class MultiAgentSignalEnvConfig(BaseModel):
    twin_config: TwinEnvConfig
    tls_ids: list[str]
    max_lanes: int = Field(gt=0)
    max_green_phases: int = Field(gt=0)
    shield_config: ShieldConfig = Field(default_factory=ShieldConfig)
    decision_interval_s: float = 5.0
    yellow_duration_s: float = 3.0
    switch_penalty: float = 1.0
    queue_norm_cap: float = 20.0
    speed_norm_cap_mps: float = 20.0

    @model_validator(mode="after")
    def _validate(self) -> MultiAgentSignalEnvConfig:
        if not self.tls_ids:
            raise ValueError("tls_ids must be non-empty")
        if self.decision_interval_s <= self.yellow_duration_s:
            raise ValueError(
                f"decision_interval_s ({self.decision_interval_s}) must be greater than "
                f"yellow_duration_s ({self.yellow_duration_s})"
            )
        return self


class MultiAgentSignalEnv:
    """PettingZoo-style (dict-in/dict-out) parallel multi-agent environment over every
    junction in `config.tls_ids`, sharing one underlying SumoTwinEnv.
    """

    def __init__(self, config: MultiAgentSignalEnvConfig) -> None:
        self._config = config

        self._mp_programs: dict[str, MaxPressureProgram] = {
            tls_id: MaxPressureProgram.from_net_file(tls_id, config.twin_config.net_file)
            for tls_id in config.tls_ids
        }
        self._incoming_lanes: dict[str, tuple[str, ...]] = {}
        self._lane_ids_for_pressure: dict[str, tuple[str, ...]] = {}
        for tls_id, program in self._mp_programs.items():
            lanes = tuple(sorted({m.from_lane for m in program.movements}))
            if len(lanes) > config.max_lanes:
                raise ValueError(
                    f"Junction '{tls_id}' has {len(lanes)} incoming lanes, exceeding "
                    f"max_lanes={config.max_lanes}. Increase max_lanes."
                )
            if len(program.green_phase_indices) > config.max_green_phases:
                raise ValueError(
                    f"Junction '{tls_id}' has {len(program.green_phase_indices)} green "
                    f"phases, exceeding max_green_phases={config.max_green_phases}."
                )
            self._incoming_lanes[tls_id] = lanes
            self._lane_ids_for_pressure[tls_id] = tuple(
                sorted(
                    {m.from_lane for m in program.movements}
                    | {m.to_lane for m in program.movements}
                )
            )

        self.agents: list[str] = list(config.tls_ids)
        # +1 for elapsed-phase-time, +1 for the corridor-coordinator hint (Week 7,
        # docs/design/04_agent_architecture.md section 4.2's "hints... consumed as
        # policy observation features"). Defaults to 0.0 for every agent until
        # set_corridor_hints() is called -- an env with no coordinator attached (every
        # env before Week 7, and every env in a corridor with no coordinator running)
        # behaves identically to before this feature existed, which is also exactly
        # what makes hint-dropout training (the roadmap's other Week 7 ask) trivial: an
        # agent trained with hints sometimes zeroed out already sees that input
        # distribution during training, not a special-cased augmentation.
        self.observation_dim = _FEATURES_PER_LANE * config.max_lanes + config.max_green_phases + 2
        self.action_dim = config.max_green_phases

        self._twin: SumoTwinEnv | None = None
        self._shields: dict[str, SignalShield] = {}
        self._corridor_hints: dict[str, float] = {}

    def reset(self, *, seed: int | None = None) -> dict[str, np.ndarray]:
        if self._twin is not None:
            self._twin.close()

        self._twin = SumoTwinEnv(self._config.twin_config)
        self._twin.reset(seed=seed)

        self._shields = {}
        for tls_id, program in self._mp_programs.items():
            initial_phase = self._twin.connection.trafficlight.getPhase(tls_id)
            signal_program = SignalProgram(tls_id, program.green_phase_indices)
            self._shields[tls_id] = SignalShield(
                signal_program, self._config.shield_config, initial_phase=initial_phase
            )

        return {tls_id: self._build_observation(tls_id) for tls_id in self.agents}

    def step(
        self, actions: dict[str, int]
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        twin = self._require_twin()
        missing = set(self.agents) - set(actions.keys())
        if missing:
            raise ValueError(f"Missing actions for agents: {sorted(missing)}")

        decisions: dict[str, tuple[int, bool, bool, str | None]] = {}
        for tls_id in self.agents:
            program = self._mp_programs[tls_id]
            shield = self._shields[tls_id]
            action = actions[tls_id] % len(program.green_phase_indices)
            requested_phase = program.green_phase_indices[action]
            decision = shield.request(requested_phase)
            phase_changed = decision.phase != shield.current_phase
            decisions[tls_id] = (decision.phase, phase_changed, decision.modified, decision.reason)

        # All junctions that need to switch execute their yellow transition together
        # (each independently timed via its own tlLogic, all sharing the same
        # yellow_duration_s), then every agent's remaining green time is stepped as one
        # shared simulation advance -- this is why the env is multi-agent over one twin,
        # not N independent twins.
        switching = [tls_id for tls_id, (_, changed, _, _) in decisions.items() if changed]
        if switching:
            for tls_id in switching:
                yellow = yellow_phase_after(
                    self._mp_programs[tls_id].phase_states, self._shields[tls_id].current_phase
                )
                twin.connection.trafficlight.setPhase(tls_id, yellow)
            twin.step(n=self._steps_for(self._config.yellow_duration_s))
            for tls_id in switching:
                target_phase = decisions[tls_id][0]
                twin.connection.trafficlight.setPhase(tls_id, target_phase)
            remaining_s = self._config.decision_interval_s - self._config.yellow_duration_s
            if remaining_s > 0:
                twin.step(n=self._steps_for(remaining_s))
        else:
            twin.step(n=self._steps_for(self._config.decision_interval_s))

        for tls_id, (target_phase, changed, _, _) in decisions.items():
            shield = self._shields[tls_id]
            if changed:
                shield.notify_phase_change(target_phase)
            shield.advance(self._config.decision_interval_s)

        observations: dict[str, np.ndarray] = {}
        rewards: dict[str, float] = {}
        terminations: dict[str, bool] = {}
        truncations: dict[str, bool] = {}
        infos: dict[str, dict[str, Any]] = {}
        for tls_id in self.agents:
            _, phase_changed, shield_modified, shield_reason = decisions[tls_id]
            pressure = total_pressure(
                self._mp_programs[tls_id], self._current_queue_lengths(tls_id)
            )
            reward = -float(pressure)
            if phase_changed:
                reward -= self._config.switch_penalty

            observations[tls_id] = self._build_observation(tls_id)
            rewards[tls_id] = reward
            terminations[tls_id] = False
            # No natural end condition, same as SingleJunctionEnv -- episode length is
            # the training script's concern (wrap with a step-count limit).
            truncations[tls_id] = False
            infos[tls_id] = {
                "pressure": pressure,
                "phase_changed": phase_changed,
                "shield_modified": shield_modified,
                "shield_reason": shield_reason,
            }
        return observations, rewards, terminations, truncations, infos

    def close(self) -> None:
        if self._twin is not None:
            self._twin.close()
            self._twin = None

    @property
    def connection(self) -> Any:
        """Escape hatch to the raw libsumo/traci handle, for exactly the kind of direct
        lane/edge-level queries this wrapper doesn't expose (e.g. an Intersection Agent
        service computing max-pressure/fixed-time fallback decisions independently of
        the RL action interface) -- the same documented pattern already used by
        `SumoTwinEnv.connection` and `SingleJunctionEnv.connection`.
        """
        return self._require_twin().connection

    @property
    def sim_time(self) -> float:
        return self._require_twin().sim_time

    def set_corridor_hints(self, hints: dict[str, float]) -> None:
        """Sets the advisory corridor-hint feature consumed by `_build_observation`
        (Week 7, docs/design/04_agent_architecture.md section 4.2). `hints` maps
        `tls_id -> hint value`; agents not present in `hints` keep whatever value they
        last had (or 0.0 if never set) -- this is a "the coordinator publishes updates
        for its own corridor's members" API, not a full-district reset every call. Pure
        advisory data: nothing about action masking, the shield, or reward computation
        reads this -- single-writer principle (docs/design/05_communication.md section
        5.1), only the observation vector.
        """
        self._corridor_hints.update(hints)

    # -- internal helpers --

    def _require_twin(self) -> SumoTwinEnv:
        if self._twin is None:
            raise RuntimeError("Environment not started. Call reset() first.")
        return self._twin

    def _steps_for(self, seconds: float) -> int:
        twin = self._require_twin()
        return max(1, round(seconds / twin.config.step_length_s))

    def _current_queue_lengths(self, tls_id: str) -> dict[str, int]:
        twin = self._require_twin()
        conn = twin.connection
        return {
            lane_id: int(conn.lane.getLastStepHaltingNumber(lane_id))
            for lane_id in self._lane_ids_for_pressure[tls_id]
        }

    def _build_observation(self, tls_id: str) -> np.ndarray:
        twin = self._require_twin()
        shield = self._shields[tls_id]
        junction_state = twin.get_junction_state(tls_id)
        lanes_by_id = {lane.lane_id: lane for lane in junction_state.incoming_lanes}
        canonical_lanes = self._incoming_lanes[tls_id]

        features: list[float] = []
        for i in range(self._config.max_lanes):
            if i < len(canonical_lanes):
                lane_id = canonical_lanes[i]
                if lane_id not in lanes_by_id:
                    raise RuntimeError(
                        f"Lane '{lane_id}' expected to be an incoming lane of '{tls_id}' "
                        "but SUMO did not report it as one."
                    )
                lane = lanes_by_id[lane_id]
                features.append(min(lane.queue_length / self._config.queue_norm_cap, 1.0))
                features.append(min(max(lane.occupancy, 0.0), 1.0))
                features.append(min(lane.mean_speed_mps / self._config.speed_norm_cap_mps, 1.0))
                features.append(1.0)  # lane_valid
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])  # padding, lane_valid=0

        program = self._mp_programs[tls_id]
        phase_one_hot = [0.0] * self._config.max_green_phases
        current_index = program.green_phase_indices.index(shield.current_phase)
        phase_one_hot[current_index] = 1.0
        features.extend(phase_one_hot)

        elapsed_norm = min(shield.elapsed_s / self._config.shield_config.max_green_s, 1.0)
        features.append(elapsed_norm)

        features.append(self._corridor_hints.get(tls_id, 0.0))

        return np.array(features, dtype=np.float32)
