"""The signal safety shield: enforces minimum/maximum green time on every signal
decision, independent of whatever produced the requested action (RL policy, max-pressure,
fixed-time, a human operator). This is the non-negotiable safety envelope described in
docs/design/03_system_architecture.md section 3.10 and 04_agent_architecture.md section
4.1 -- "the shield is unit-tested as a standalone library and wired into both training
(masked action distributions) and execution identically, so there's no train/serve skew."

Deliberately SUMO-agnostic: this module never talks to libsumo/traci. It is a pure
decision-time gatekeeper over (phase, elapsed_s) state that the caller reports in and
gets validated decisions back from. Wiring it to an actual running SUMO simulation
(applying the chosen phase, executing the yellow transition between green phases) is a
separate, later concern -- the Intersection Agent runtime (Week 6 per the roadmap), not
this module. Keeping the shield decoupled from SUMO is what makes it fast to unit-test
exhaustively (see tests/unit/test_shield.py's property-based tests) without needing a
live simulation for every case.

Phase model, grounded in the real network's tlLogic definitions (verified before writing
this): a junction's SUMO phases alternate between controllable "green" phases (arbitrary
mixes of 'r'/'G'/'g', no 'y'/'Y') and short "yellow" transition phases (contain 'y'/'Y').
Only green phases are ever legal targets for a requested action -- yellow phases are
executor-managed transitions, never chosen by a policy or controller, matching the
PressLight/CoLight/MPLight convention this project's control plane is built on
(docs/design/02_literature_survey.md section 2.4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from atlas.control.phases import green_phase_indices as _green_phase_indices

ShieldReason = Literal["min_green", "max_green", "invalid_phase", "preemption_lock"]


@dataclass(frozen=True, slots=True)
class SignalProgram:
    """The static, immutable set of controllable (green) phases for one junction,
    derived from its SUMO tlLogic phase states."""

    tls_id: str
    green_phase_indices: tuple[int, ...]
    """SUMO phase indices, in the tlLogic's original cyclic order."""

    @classmethod
    def from_phase_states(cls, tls_id: str, phase_states: list[str]) -> SignalProgram:
        green = _green_phase_indices(phase_states)
        if not green:
            raise ValueError(f"No green (non-yellow) phases found for traffic light '{tls_id}'")
        return cls(tls_id=tls_id, green_phase_indices=green)

    def next_green(self, current_phase: int) -> int:
        """The next green phase after `current_phase`, cycling back to the first.
        The shield's fallback when a max-green violation forces a switch but the caller
        didn't (couldn't, in the masked-RL case) specify which phase to switch to."""
        idx = self.green_phase_indices.index(current_phase)
        return self.green_phase_indices[(idx + 1) % len(self.green_phase_indices)]


class ShieldConfig(BaseModel):
    min_green_s: float = Field(default=10.0, gt=0)
    max_green_s: float = Field(default=60.0, gt=0)

    @model_validator(mode="after")
    def _validate_bounds(self) -> ShieldConfig:
        if self.max_green_s <= self.min_green_s:
            raise ValueError(
                f"max_green_s ({self.max_green_s}) must be greater than "
                f"min_green_s ({self.min_green_s})"
            )
        return self


@dataclass(frozen=True, slots=True)
class ShieldDecision:
    """The outcome of validating one requested action: the phase to actually apply, and
    whether/why it differs from what was requested -- mirrors the `shield: {modified,
    reason}` fields on the `control.commands.v1` event schema (Part 5 of the design docs),
    so this maps directly onto the audit log once the control plane is wired to the bus.
    """

    phase: int
    modified: bool
    reason: ShieldReason | None


class SignalShield:
    """Stateful per-junction safety shield. Tracks how long the current phase has held
    and validates every requested action against min/max green, guaranteeing by
    construction -- not by trusting the caller -- that no returned decision ever violates
    those bounds.
    """

    def __init__(
        self,
        program: SignalProgram,
        config: ShieldConfig,
        initial_phase: int | None = None,
    ) -> None:
        self._program = program
        self._config = config
        self._current_phase = (
            initial_phase if initial_phase is not None else program.green_phase_indices[0]
        )
        if self._current_phase not in program.green_phase_indices:
            raise ValueError(
                f"initial_phase {self._current_phase} is not a green phase for '{program.tls_id}'"
            )
        self._elapsed_s = 0.0

    @property
    def current_phase(self) -> int:
        return self._current_phase

    @property
    def elapsed_s(self) -> float:
        return self._elapsed_s

    def legal_actions(self) -> frozenset[int]:
        """The set of green phase indices a caller may legally request right now -- what
        an RL policy's action distribution should be masked to (Part 6.1's "action
        masking encodes the safety shield").
        """
        if self._elapsed_s < self._config.min_green_s:
            return frozenset({self._current_phase})
        if self._elapsed_s >= self._config.max_green_s:
            return frozenset(self._program.green_phase_indices) - {self._current_phase}
        return frozenset(self._program.green_phase_indices)

    def request(self, requested_phase: int) -> ShieldDecision:
        """Validate a requested phase against current shield state. Does not mutate
        internal state -- call `advance`/`notify_phase_change` separately once the
        decision has actually been applied, so a caller can evaluate a hypothetical
        request without committing to it.

        An invalid phase request is treated as if "keep current phase" had been
        requested for the purposes of min/max-green enforcement -- it must NOT be a way
        to silently bypass the max-green override (a real bug caught by the property
        test in tests/unit/test_shield.py: an invalid request arriving exactly at
        max_green_s used to fall through as an unchecked "keep").
        """
        is_valid = requested_phase in self._program.green_phase_indices
        effective_requested = requested_phase if is_valid else self._current_phase
        wants_switch = effective_requested != self._current_phase

        if wants_switch and self._elapsed_s < self._config.min_green_s:
            return ShieldDecision(phase=self._current_phase, modified=True, reason="min_green")

        if not wants_switch and self._elapsed_s >= self._config.max_green_s:
            forced = self._program.next_green(self._current_phase)
            return ShieldDecision(phase=forced, modified=True, reason="max_green")

        if not is_valid:
            return ShieldDecision(phase=self._current_phase, modified=True, reason="invalid_phase")

        return ShieldDecision(phase=effective_requested, modified=False, reason=None)

    def force(self, phase: int) -> ShieldDecision:
        """Unconditionally applies `phase`, bypassing min/max-green -- the Preemption
        Overlay's mechanism (docs/design/04_agent_architecture.md section 4.3: "command
        green waves... phase to the approach's green with standard transition") for
        commanding a junction during an active emergency lock, when the normal timing
        envelope must yield to a higher-priority safety concern. Still refuses a phase
        that isn't one of this junction's real green phases -- overriding the timing
        envelope is intentional; letting a bogus phase index through never is.
        """
        if phase not in self._program.green_phase_indices:
            raise ValueError(f"{phase} is not a green phase for '{self._program.tls_id}'")
        modified = phase != self._current_phase
        return ShieldDecision(
            phase=phase, modified=modified, reason="preemption_lock" if modified else None
        )

    def advance(self, seconds: float) -> None:
        """Call once a decision has been applied and the simulation has moved forward by
        `seconds` while remaining on the same phase."""
        if seconds < 0:
            raise ValueError(f"seconds must be >= 0, got {seconds}")
        self._elapsed_s += seconds

    def notify_phase_change(self, new_phase: int) -> None:
        """Call once the executor has actually switched to `new_phase` (after any yellow
        transition has run), resetting the elapsed-time clock."""
        if new_phase not in self._program.green_phase_indices:
            raise ValueError(f"{new_phase} is not a green phase for '{self._program.tls_id}'")
        self._current_phase = new_phase
        self._elapsed_s = 0.0
