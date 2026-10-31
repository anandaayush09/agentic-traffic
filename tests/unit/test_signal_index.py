"""Validates the parse-once signal index and the city-scale simulation options.

The index replaces "re-parse the whole network file for every junction" (hours at borough scale)
with one streaming pass, so the property that matters is EQUIVALENCE: for every real signal it must
produce exactly what the per-junction functions produce, otherwise controllers would silently behave
differently. That is asserted against the real Midtown district, signal by signal.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from atlas.common.config import project_root
from atlas.control.fixed_time import native_green_durations_from_net_file
from atlas.control.max_pressure import MaxPressureProgram
from atlas.control.signal_index import NetworkSignalIndex
from atlas.twin import SumoTwinEnv, TwinEnvConfig
from atlas.twin.controlled_twin import DISTRICT_TLS_IDS, ControlledTwin

NETWORK_DIR = project_root() / "data" / "networks" / "manhattan_midtown"
NET_FILE = NETWORK_DIR / "net.net.xml"
ROUTE_FILE = NETWORK_DIR / "demand" / "pm_peak.rou.xml"
VTYPES_FILE = NETWORK_DIR / "vtypes.add.xml"

requires_network = pytest.mark.skipif(not NET_FILE.is_file(), reason="manhattan_midtown not built")

TINY = """<net>
  <tlLogic id="ok" type="static"><phase duration="30" state="GGrr"/><phase duration="3" state="yyrr"/>
    <phase duration="25" state="rrGG"/><phase duration="3" state="rryy"/></tlLogic>
  <tlLogic id="nogreen" type="static"><phase duration="10" state="rr"/></tlLogic>
  <tlLogic id="noconn" type="static"><phase duration="30" state="GG"/></tlLogic>
  <connection from="a" to="b" fromLane="0" toLane="0" tl="ok" linkIndex="0"/>
  <connection from="a" to="b" fromLane="1" toLane="1" tl="ok" linkIndex="1"/>
  <connection from="c" to="b" fromLane="0" toLane="0" tl="ok" linkIndex="2"/>
  <connection from="a" to="b" fromLane="0" toLane="0" tl="nogreen" linkIndex="0"/>
  <connection from="x" to="y" fromLane="0" toLane="0" tl="nolog" linkIndex="0"/>
</net>"""


@pytest.fixture
def tiny(tmp_path: Path) -> Path:
    path = tmp_path / "tiny.net.xml"
    path.write_text(TINY, encoding="utf-8")
    return path


class TestSemantics:
    def test_a_controllable_signal_gets_its_program_durations_and_lane_count(
        self, tiny: Path
    ) -> None:
        index = NetworkSignalIndex.from_net_file(tiny)

        program = index.program("ok")
        assert program.green_phase_indices == (0, 2)
        assert [m.from_lane for m in program.movements] == ["a_0", "a_1", "c_0"]
        assert index.native_durations("ok") == ((0, 2), (30.0, 25.0))
        assert index.required_max_lanes(["ok"]) == 3  # a_0, a_1, c_0
        assert index.required_max_green_phases(["ok"]) == 2

    def test_signals_atlas_cannot_control_are_skipped_with_the_reason(self, tiny: Path) -> None:
        index = NetworkSignalIndex.from_net_file(tiny)

        assert index.controllable_ids == ["ok"]
        assert index.skipped == {
            "nogreen": "no controllable green phase",
            "noconn": "no controlled connections",
            "nolog": "no signal program (tlLogic)",
        }

    def test_asking_to_control_an_uncontrollable_signal_names_it_and_why(self, tiny: Path) -> None:
        index = NetworkSignalIndex.from_net_file(tiny)

        with pytest.raises(ValueError, match=r"'nogreen' \(no controllable green phase\)"):
            index.require(["ok", "nogreen"])
        with pytest.raises(ValueError, match=r"'ghost' \(not in the network\)"):
            index.require(["ghost"])
        index.require(["ok"])  # fine

    def test_restricting_to_some_ids_ignores_the_rest_without_calling_them_skipped(
        self, tiny: Path
    ) -> None:
        index = NetworkSignalIndex.from_net_file(tiny, tls_ids=["ok"])

        assert index.controllable_ids == ["ok"]
        assert index.skipped == {}

    def test_the_first_program_for_an_id_wins_like_the_per_junction_lookup(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "two.net.xml"
        path.write_text(
            """<net><tlLogic id="t" programID="0"><phase duration="11" state="Gr"/><phase duration="3" state="yr"/></tlLogic>
            <tlLogic id="t" programID="1"><phase duration="99" state="Gr"/></tlLogic>
            <connection from="a" to="b" fromLane="0" toLane="0" tl="t" linkIndex="0"/></net>""",
            encoding="utf-8",
        )

        assert NetworkSignalIndex.from_net_file(path).native_durations("t") == ((0,), (11.0,))


@requires_network
class TestEquivalenceWithThePerJunctionFunctions:
    index = NetworkSignalIndex.from_net_file(NET_FILE) if NET_FILE.is_file() else None

    @pytest.mark.parametrize("tls_id", DISTRICT_TLS_IDS)
    def test_the_program_is_identical_for_every_real_signal(self, tls_id: str) -> None:
        assert self.index is not None

        assert self.index.program(tls_id) == MaxPressureProgram.from_net_file(tls_id, NET_FILE)

    @pytest.mark.parametrize("tls_id", DISTRICT_TLS_IDS)
    def test_the_native_durations_are_identical_for_every_real_signal(self, tls_id: str) -> None:
        assert self.index is not None

        assert self.index.native_durations(tls_id) == native_green_durations_from_net_file(
            tls_id, NET_FILE
        )

    def test_the_district_needs_exactly_the_sizes_it_always_hard_coded(self) -> None:
        assert self.index is not None

        assert len(self.index) == 22
        assert self.index.required_max_lanes() == 10
        assert self.index.required_max_green_phases() == 2


class _FakeSumo:
    """Records the command line SumoTwinEnv would launch SUMO with."""

    def __init__(self) -> None:
        self.cmd: list[str] = []
        self.simulation: Any = type("S", (), {"getTime": staticmethod(lambda: 0.0)})()

    def start(self, cmd: list[str]) -> None:
        self.cmd = cmd

    def close(self) -> None:
        pass


class TestSimulationOptions:
    def _command(self, monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> list[str]:
        fake = _FakeSumo()
        config = TwinEnvConfig(name="t", net_file=Path("n.xml"), route_file=Path("r.xml"), **kwargs)
        env = SumoTwinEnv(config)
        monkeypatch.setattr(env, "_load_backend", lambda: fake)
        monkeypatch.setattr("atlas.twin.env.get_binary_path", lambda name: Path(name))
        env.reset()
        return fake.cmd

    def test_defaults_leave_the_district_exactly_as_it_was(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cmd = self._command(monkeypatch)

        assert cmd[cmd.index("--time-to-teleport") + 1] == "-1"
        assert "--scale" not in cmd

    def test_a_city_can_enable_teleporting_so_gridlock_cannot_freeze_it_forever(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cmd = self._command(monkeypatch, time_to_teleport_s=300)

        assert float(cmd[cmd.index("--time-to-teleport") + 1]) == 300.0

    def test_demand_can_be_scaled_without_regenerating_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cmd = self._command(monkeypatch, scale=0.25)

        assert cmd[cmd.index("--scale") + 1] == "0.25"

    def test_a_non_positive_scale_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            TwinEnvConfig(name="t", net_file=Path("n"), route_file=Path("r"), scale=0)


@requires_network
class TestControlledTwinUsesTheIndex:
    def test_the_district_keeps_its_exact_shapes(self) -> None:
        twin = ControlledTwin(NET_FILE, ROUTE_FILE, [VTYPES_FILE])

        assert (twin._max_lanes, twin._max_green_phases) == (10, 2)
        assert twin.tls_ids == list(DISTRICT_TLS_IDS)

    def test_sizes_are_derived_upward_when_a_network_needs_more(self, tmp_path: Path) -> None:
        index = NetworkSignalIndex.from_net_file(NET_FILE)
        wide = NetworkSignalIndex({**{t: index._signals[t] for t in index.controllable_ids}}, {})
        assert wide.required_max_lanes() == 10  # sanity: the real network, unchanged

        # A hypothetical 26-lane, 7-phase signal (what Manhattan's audit found) must raise the
        # derived sizes rather than overflow the fixed ones.
        from dataclasses import replace

        big_program = replace(
            index.program("42430333"),
            movements=tuple(
                replace(index.program("42430333").movements[0], from_lane=f"e_{i}")
                for i in range(26)
            ),
            green_phase_indices=tuple(range(7)),
            phase_states=tuple("G" * 4 for _ in range(7)),
        )
        from atlas.control.signal_index import SignalInfo

        signals = dict(index._signals)
        signals["42430333"] = SignalInfo(big_program, ((0,), (30.0,)), 26)
        twin = ControlledTwin(
            NET_FILE, ROUTE_FILE, [VTYPES_FILE], signal_index=NetworkSignalIndex(signals, {})
        )

        assert (twin._max_lanes, twin._max_green_phases) == (26, 7)

    def test_all_means_every_controllable_signal_in_the_network(self) -> None:
        twin = ControlledTwin(NET_FILE, ROUTE_FILE, [VTYPES_FILE], tls_ids="all")

        assert len(twin.tls_ids) == 22

    def test_an_uncontrollable_signal_is_a_clear_error_at_construction(self) -> None:
        with pytest.raises(ValueError, match="Cannot control: 'nonexistent'"):
            ControlledTwin(NET_FILE, ROUTE_FILE, [VTYPES_FILE], tls_ids=["nonexistent"])
