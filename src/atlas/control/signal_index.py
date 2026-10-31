"""NetworkSignalIndex: every signal's control structure, read from the network file ONCE.

`MaxPressureProgram.from_net_file` and `native_green_durations_from_net_file` each parse the entire
network XML to extract ONE junction. Called once per controlled junction that is harmless for a
22-signal district (a 0.2 MB file) and ruinous for a borough: Manhattan has 2,670 signals in a 40 MB
network, so the per-junction pattern means 2,670 x 2 full parses -- hours of startup before the first
simulation step. This module does the same extraction in a single streaming pass for all signals.

It builds exactly the objects the per-junction functions build (same movements in the same order,
same phase strings, same green-phase rule from `atlas.control.phases`), which the tests assert
signal by signal against the originals on the Midtown district -- so nothing downstream changes, only
how many times the file is read. The per-junction functions remain, for single-junction use.

It also answers the sizing question a city forces: which `max_lanes` / `max_green_phases` the
policy observation and action shapes need so that no controlled junction overflows them.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

from atlas.control.max_pressure import MaxPressureProgram, Movement
from atlas.control.phases import green_phase_indices


@dataclass(frozen=True, slots=True)
class SignalInfo:
    program: MaxPressureProgram
    native_durations: tuple[tuple[int, ...], tuple[float, ...]]
    """(green_phase_indices, durations_s): the network's own fixed-time plan, as
    `native_green_durations_from_net_file` returns it."""
    incoming_lanes: int
    """Distinct lanes feeding the junction's controlled movements."""


class NetworkSignalIndex:
    def __init__(self, signals: dict[str, SignalInfo], skipped: dict[str, str]) -> None:
        self._signals = signals
        self._skipped = skipped

    @classmethod
    def from_net_file(
        cls, net_file: Path, tls_ids: Collection[str] | None = None
    ) -> NetworkSignalIndex:
        """One streaming pass over `net_file`. With `tls_ids`, only those signals are indexed
        (others are ignored, not reported as skipped); without, every signal in the network."""
        wanted = set(tls_ids) if tls_ids is not None else None
        movements: dict[str, list[Movement]] = {}
        logics: dict[str, list[tuple[str, float]]] = {}

        for _, element in ElementTree.iterparse(net_file, events=("end",)):
            tag = element.tag
            if tag == "connection":
                tls = element.get("tl")
                if tls and (wanted is None or tls in wanted):
                    movements.setdefault(tls, []).append(
                        Movement(
                            link_index=int(element.get("linkIndex", "-1")),
                            from_lane=f"{element.get('from')}_{element.get('fromLane')}",
                            to_lane=f"{element.get('to')}_{element.get('toLane')}",
                        )
                    )
                element.clear()
            elif tag == "tlLogic":
                tls = element.get("id", "")
                # The first program for an id wins, as in the per-junction lookup.
                if (wanted is None or tls in wanted) and tls not in logics:
                    logics[tls] = [
                        (p.get("state", ""), float(p.get("duration", "0")))
                        for p in element.findall("phase")
                    ]
                element.clear()

        signals: dict[str, SignalInfo] = {}
        skipped: dict[str, str] = {}
        ids = wanted if wanted is not None else set(logics) | set(movements)
        for tls in sorted(ids):
            if tls not in movements:
                skipped[tls] = "no controlled connections"
                continue
            if tls not in logics:
                skipped[tls] = "no signal program (tlLogic)"
                continue
            states = tuple(state for state, _ in logics[tls])
            greens = green_phase_indices(states)
            if not greens:
                skipped[tls] = "no controllable green phase"
                continue
            durations = tuple(logics[tls][i][1] for i in greens)
            signals[tls] = SignalInfo(
                program=MaxPressureProgram(
                    tls_id=tls,
                    movements=tuple(movements[tls]),
                    green_phase_indices=greens,
                    phase_states=states,
                ),
                native_durations=(greens, durations),
                incoming_lanes=len({m.from_lane for m in movements[tls]}),
            )
        return cls(signals, skipped)

    # -- queries --

    @property
    def controllable_ids(self) -> list[str]:
        return sorted(self._signals)

    @property
    def skipped(self) -> dict[str, str]:
        """Signal id -> why ATLAS can't control it (it keeps SUMO's own program)."""
        return dict(self._skipped)

    def __contains__(self, tls_id: object) -> bool:
        return tls_id in self._signals

    def __len__(self) -> int:
        return len(self._signals)

    def require(self, tls_ids: Sequence[str]) -> None:
        """Raises a clear error naming every id that can't be controlled and why."""
        problems = [
            f"'{t}' ({self._skipped.get(t, 'not in the network')})"
            for t in tls_ids
            if t not in self._signals
        ]
        if problems:
            raise ValueError(f"Cannot control: {', '.join(problems)}")

    def program(self, tls_id: str) -> MaxPressureProgram:
        return self._signals[tls_id].program

    def programs(self, tls_ids: Sequence[str] | None = None) -> dict[str, MaxPressureProgram]:
        ids = self.controllable_ids if tls_ids is None else tls_ids
        return {t: self._signals[t].program for t in ids}

    def native_durations(self, tls_id: str) -> tuple[tuple[int, ...], tuple[float, ...]]:
        return self._signals[tls_id].native_durations

    def required_max_lanes(self, tls_ids: Sequence[str] | None = None) -> int:
        ids = self.controllable_ids if tls_ids is None else tls_ids
        return max((self._signals[t].incoming_lanes for t in ids), default=0)

    def required_max_green_phases(self, tls_ids: Sequence[str] | None = None) -> int:
        ids = self.controllable_ids if tls_ids is None else tls_ids
        return max((len(self._signals[t].program.green_phase_indices) for t in ids), default=0)
