"""SingleJunctionEnv: a Gymnasium environment wrapping SumoTwinEnv to control ONE
junction's signal phases, for the Week 3 RL ramp-up (docs/design/13_roadmap.md, Week 3;
docs/design/06_ai_pipeline.md section 6.1). This is deliberately the smallest possible
slice of the eventual multi-agent control plane -- one agent, one junction, no corridor
coordination, no bus -- built to develop RL intuition against a known-good library
(Stable-Baselines3) before the from-scratch MAPPO implementation (Week 5).

Every design choice here is a direct instance of something docs/design/06_ai_pipeline.md
section 6.1 specifies, dogfooding modules already built rather than reinventing them:

- **Action space**: `Discrete(num_green_phases)`. The agent never sees or picks yellow
  phases -- see atlas.control.phases. An action is a *request*; it is validated by a real
  `SignalShield` before being applied, exactly as production execution will do (Part
  4.1's "no train/serve skew" -- the shield code path is identical here and in real
  control, not reimplemented).
- **Observation**: per incoming lane [queue_norm, occupancy, speed_norm] + current-phase
  one-hot + normalized elapsed-time-in-phase, all in [0, 1].
- **Reward**: negative whole-junction pressure (`atlas.control.max_pressure.
  total_pressure` -- the exact function real max-pressure control also uses to choose
  phases, reused here as a reward rather than a decision rule) plus a small penalty for
  switching phases, encouraging stability over thrashing.
- **Decision cadence**: every `decision_interval_s` (default 5s, matching Part 6.1). A
  phase switch is executed with a real yellow transition (via
  `atlas.control.phases.yellow_phase_after`), not an instantaneous unsafe jump.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from pydantic import BaseModel, Field

from atlas.control.max_pressure import MaxPressureProgram, total_pressure
from atlas.control.phases import yellow_phase_after
from atlas.control.shield import ShieldConfig, SignalProgram, SignalShield
from atlas.twin import SumoTwinEnv, TwinEnvConfig

_FEATURES_PER_LANE = 3  # queue_norm, occupancy, speed_norm


class SingleJunctionEnvConfig(BaseModel):
    twin_config: TwinEnvConfig
    tls_id: str
    shield_config: ShieldConfig = Field(default_factory=ShieldConfig)
    decision_interval_s: float = 5.0
    yellow_duration_s: float = 3.0
    switch_penalty: float = 1.0
    queue_norm_cap: float = 20.0
    speed_norm_cap_mps: float = 20.0


class SingleJunctionEnv(gym.Env[np.ndarray, np.integer]):
    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(self, config: SingleJunctionEnvConfig) -> None:
        super().__init__()
        if config.decision_interval_s <= config.yellow_duration_s:
            raise ValueError(
                f"decision_interval_s ({config.decision_interval_s}) must be greater "
                f"than yellow_duration_s ({config.yellow_duration_s})"
            )
        self._config = config

        self._mp_program = MaxPressureProgram.from_net_file(
            config.tls_id, config.twin_config.net_file
        )
        self._incoming_lanes: tuple[str, ...] = tuple(
            sorted({m.from_lane for m in self._mp_program.movements})
        )
        self._num_green = len(self._mp_program.green_phase_indices)
        self._lane_ids_for_pressure: tuple[str, ...] = tuple(
            sorted(
                {m.from_lane for m in self._mp_program.movements}
                | {m.to_lane for m in self._mp_program.movements}
            )
        )

        self.action_space = gym.spaces.Discrete(self._num_green)
        obs_dim = _FEATURES_PER_LANE * len(self._incoming_lanes) + self._num_green + 1
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        self._twin: SumoTwinEnv | None = None
        self._shield: SignalShield | None = None

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if self._twin is not None:
            self._twin.close()

        self._twin = SumoTwinEnv(self._config.twin_config)
        self._twin.reset(seed=seed)

        program = SignalProgram(self._config.tls_id, self._mp_program.green_phase_indices)
        initial_phase = self._twin.connection.trafficlight.getPhase(self._config.tls_id)
        self._shield = SignalShield(
            program, self._config.shield_config, initial_phase=initial_phase
        )

        return self._build_observation(), {}

    def step(self, action: np.integer) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        twin, shield = self._require_started()

        requested_phase = self._mp_program.green_phase_indices[int(action)]
        decision = shield.request(requested_phase)
        phase_changed = decision.phase != shield.current_phase

        if phase_changed:
            self._execute_phase_change(decision.phase)
            shield.notify_phase_change(decision.phase)
            remaining_s = self._config.decision_interval_s - self._config.yellow_duration_s
            if remaining_s > 0:
                twin.step(n=self._steps_for(remaining_s))
            shield.advance(self._config.decision_interval_s)
        else:
            twin.step(n=self._steps_for(self._config.decision_interval_s))
            shield.advance(self._config.decision_interval_s)

        pressure = total_pressure(self._mp_program, self._current_queue_lengths())
        reward = -float(pressure)
        if phase_changed:
            reward -= self._config.switch_penalty

        observation = self._build_observation()
        # This environment has no natural end condition -- SUMO keeps stepping an
        # (eventually empty) network indefinitely even after all demand has departed and
        # arrived, so there is nothing genuine to signal here. Episode length is the
        # training script's concern: wrap with `gymnasium.wrappers.TimeLimit`.
        terminated = False
        truncated = False
        info: dict[str, Any] = {
            "pressure": pressure,
            "phase_changed": phase_changed,
            "shield_modified": decision.modified,
            "shield_reason": decision.reason,
        }
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        if self._twin is not None:
            self._twin.close()
            self._twin = None

    @property
    def connection(self) -> Any:
        """The raw libsumo/traci handle, for callers that need to compute actions from
        live simulation state this environment doesn't itself expose -- e.g. an
        evaluation script running a classical controller (max-pressure, fixed-time)
        through this same env for a fair side-by-side comparison against a trained
        policy. Mirrors SumoTwinEnv.connection's escape-hatch precedent exactly.
        """
        twin, _ = self._require_started()
        return twin.connection

    # -- internal helpers --

    def _require_started(self) -> tuple[SumoTwinEnv, SignalShield]:
        if self._twin is None or self._shield is None:
            raise RuntimeError("Environment not started. Call reset() first.")
        return self._twin, self._shield

    def _steps_for(self, seconds: float) -> int:
        assert self._twin is not None
        return max(1, round(seconds / self._twin.config.step_length_s))

    def _execute_phase_change(self, target_green_phase: int) -> None:
        twin, shield = self._require_started()
        yellow = yellow_phase_after(self._mp_program.phase_states, shield.current_phase)
        conn = twin.connection
        conn.trafficlight.setPhase(self._config.tls_id, yellow)
        twin.step(n=self._steps_for(self._config.yellow_duration_s))
        conn.trafficlight.setPhase(self._config.tls_id, target_green_phase)

    def _current_queue_lengths(self) -> dict[str, int]:
        twin, _ = self._require_started()
        conn = twin.connection
        return {
            lane_id: int(conn.lane.getLastStepHaltingNumber(lane_id))
            for lane_id in self._lane_ids_for_pressure
        }

    def _build_observation(self) -> np.ndarray:
        twin, shield = self._require_started()
        junction_state = twin.get_junction_state(self._config.tls_id)
        lanes_by_id = {lane.lane_id: lane for lane in junction_state.incoming_lanes}

        features: list[float] = []
        for lane_id in self._incoming_lanes:
            if lane_id not in lanes_by_id:
                raise RuntimeError(
                    f"Lane '{lane_id}' expected to be an incoming lane of "
                    f"'{self._config.tls_id}' but SUMO did not report it as one."
                )
            lane = lanes_by_id[lane_id]
            features.append(min(lane.queue_length / self._config.queue_norm_cap, 1.0))
            features.append(min(max(lane.occupancy, 0.0), 1.0))
            features.append(min(lane.mean_speed_mps / self._config.speed_norm_cap_mps, 1.0))

        phase_one_hot = [0.0] * self._num_green
        current_index_in_green = self._mp_program.green_phase_indices.index(shield.current_phase)
        phase_one_hot[current_index_in_green] = 1.0
        features.extend(phase_one_hot)

        elapsed_norm = min(shield.elapsed_s / self._config.shield_config.max_green_s, 1.0)
        features.append(elapsed_norm)

        return np.array(features, dtype=np.float32)
