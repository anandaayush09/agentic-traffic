"""Validates the signal safety shield. Two layers: concrete example tests documenting
expected behavior for readability, and property-based tests (Hypothesis) proving the
timing invariants hold across thousands of randomly generated request sequences, not just
the handful of cases a human thinks to write by hand -- this is what the Week 3 roadmap
means by "shield invariant tests (property-based)".
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from atlas.control.shield import ShieldConfig, ShieldDecision, SignalProgram, SignalShield

# Phase states copied verbatim from a real junction in the built network (42428283),
# grounding the tests in real SUMO output rather than an invented example.
REAL_PHASE_STATES = ["GGrr", "yyrr", "rrGG", "rryy"]


class TestSignalProgram:
    def test_identifies_green_phases_from_real_states(self) -> None:
        program = SignalProgram.from_phase_states("42428283", REAL_PHASE_STATES)

        assert program.green_phase_indices == (0, 2)

    def test_rejects_all_yellow_states(self) -> None:
        with pytest.raises(ValueError, match="No green"):
            SignalProgram.from_phase_states("bad", ["yyyy", "YYYY"])

    def test_next_green_cycles(self) -> None:
        program = SignalProgram("x", (0, 2, 5))

        assert program.next_green(0) == 2
        assert program.next_green(2) == 5
        assert program.next_green(5) == 0  # wraps around


class TestShieldConfig:
    def test_rejects_max_not_greater_than_min(self) -> None:
        with pytest.raises(ValidationError, match="must be greater than"):
            ShieldConfig(min_green_s=30.0, max_green_s=30.0)


class TestSignalShieldExamples:
    def _make_shield(self, min_green_s: float = 10.0, max_green_s: float = 60.0) -> SignalShield:
        program = SignalProgram("test", (0, 1, 2))
        return SignalShield(program, ShieldConfig(min_green_s=min_green_s, max_green_s=max_green_s))

    def test_keep_is_always_legal_immediately_after_switch(self) -> None:
        shield = self._make_shield()

        decision = shield.request(shield.current_phase)

        assert decision == ShieldDecision(phase=shield.current_phase, modified=False, reason=None)

    def test_early_switch_request_is_blocked(self) -> None:
        shield = self._make_shield(min_green_s=10.0)
        shield.advance(5.0)  # less than min_green

        decision = shield.request(1)  # currently on phase 0

        assert decision.modified is True
        assert decision.reason == "min_green"
        assert decision.phase == 0  # forced to keep

    def test_switch_after_min_green_is_allowed(self) -> None:
        shield = self._make_shield(min_green_s=10.0)
        shield.advance(10.0)

        decision = shield.request(1)

        assert decision == ShieldDecision(phase=1, modified=False, reason=None)

    def test_keep_past_max_green_is_forced_to_switch(self) -> None:
        shield = self._make_shield(max_green_s=60.0)
        shield.advance(60.0)

        decision = shield.request(shield.current_phase)  # tries to keep

        assert decision.modified is True
        assert decision.reason == "max_green"
        assert decision.phase == 1  # program.next_green(0) == 1

    def test_invalid_phase_request_is_rejected(self) -> None:
        shield = self._make_shield()

        decision = shield.request(99)

        assert decision.modified is True
        assert decision.reason == "invalid_phase"
        assert decision.phase == shield.current_phase

    def test_notify_phase_change_resets_elapsed(self) -> None:
        shield = self._make_shield(min_green_s=10.0)
        shield.advance(10.0)

        shield.notify_phase_change(1)

        assert shield.current_phase == 1
        assert shield.elapsed_s == 0.0

    def test_notify_phase_change_rejects_non_green_phase(self) -> None:
        shield = self._make_shield()

        with pytest.raises(ValueError, match="not a green phase"):
            shield.notify_phase_change(99)

    def test_advance_rejects_negative_seconds(self) -> None:
        shield = self._make_shield()

        with pytest.raises(ValueError, match="seconds must be"):
            shield.advance(-1.0)

    def test_legal_actions_restricted_to_current_before_min_green(self) -> None:
        shield = self._make_shield(min_green_s=10.0)
        shield.advance(3.0)

        assert shield.legal_actions() == frozenset({0})

    def test_legal_actions_excludes_current_after_max_green(self) -> None:
        shield = self._make_shield(max_green_s=60.0)
        shield.advance(60.0)

        assert shield.legal_actions() == frozenset({1, 2})

    def test_legal_actions_all_phases_in_normal_window(self) -> None:
        shield = self._make_shield(min_green_s=10.0, max_green_s=60.0)
        shield.advance(30.0)

        assert shield.legal_actions() == frozenset({0, 1, 2})


# -- Property-based invariant tests --

_PHASES = (0, 1, 2, 3)


def _apply(shield: SignalShield, decision: ShieldDecision) -> None:
    if decision.phase != shield.current_phase:
        shield.notify_phase_change(decision.phase)


@given(
    min_green=st.floats(min_value=1.0, max_value=30.0, allow_nan=False),
    max_green_offset=st.floats(min_value=1.0, max_value=90.0, allow_nan=False),
    events=st.lists(
        st.tuples(
            st.floats(min_value=0.0, max_value=20.0, allow_nan=False),  # seconds to advance
            st.integers(min_value=-1, max_value=4),  # requested phase, some invalid
        ),
        min_size=1,
        max_size=100,
    ),
)
@settings(max_examples=200)
def test_shield_never_violates_min_or_max_green(
    min_green: float, max_green_offset: float, events: list[tuple[float, int]]
) -> None:
    """The property the Week 3 roadmap calls for: no shield output, across any sequence
    of requests, ever violates min/max green -- and every returned phase is always a
    legitimate green phase for the program."""
    max_green = min_green + max_green_offset  # guarantees max > min, per ShieldConfig
    program = SignalProgram("prop_test", _PHASES)
    config = ShieldConfig(min_green_s=min_green, max_green_s=max_green)
    shield = SignalShield(program, config)

    for advance_s, requested_phase in events:
        elapsed_before = shield.elapsed_s
        phase_before = shield.current_phase

        shield.advance(advance_s)
        decision = shield.request(requested_phase)

        # Invariant: output is always a real green phase.
        assert decision.phase in _PHASES

        switched = decision.phase != phase_before
        total_elapsed = elapsed_before + advance_s

        if switched:
            # Invariant: a switch never happens before min_green has elapsed on the
            # phase being left.
            assert total_elapsed >= min_green
        else:
            # Invariant: "keep" is never returned once max_green has been reached.
            assert total_elapsed < max_green

        _apply(shield, decision)
