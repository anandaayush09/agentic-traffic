"""Validates the live-traffic service: the pure incident-to-disruption mapping, and
(against the real Manhattan network + pm_peak demand, the same inputs the demo API uses)
that the background twin really produces geometry, vehicles and per-road speeds, and that
an incident really becomes a scenario-engine disruption on the roads it names.

The simulation tests drive `start_simulation()`/`tick()` directly instead of the
background thread so they are deterministic; one test exercises the thread itself.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

import pytest

from atlas.common.config import project_root
from atlas.services.live_traffic import LiveTrafficService, plan_disruption_specs

NETWORK_DIR = project_root() / "data" / "networks" / "manhattan_midtown"
NET_FILE = NETWORK_DIR / "net.net.xml"
ROUTE_FILE = NETWORK_DIR / "demand" / "pm_peak.rou.xml"
VTYPES_FILE = NETWORK_DIR / "vtypes.add.xml"

# Midtown, with margin -- every geometry point and vehicle must land inside it.
LON_RANGE = (-74.02, -73.95)
LAT_RANGE = (40.74, 40.77)

# A corridor junction with two incoming roads (verified against the built network).
JUNCTION = "42430342"
JUNCTION_INCOMING = {"1117867028", "127693103#3"}

requires_network = pytest.mark.skipif(
    not (NET_FILE.is_file() and ROUTE_FILE.is_file()),
    reason="manhattan_midtown network not built (run scripts/build_district_network.py "
    "and scripts/generate_demand.py)",
)


def _make_service(**overrides: Any) -> LiveTrafficService:
    params: dict[str, Any] = {"warmup_s": 60}
    params.update(overrides)
    return LiveTrafficService(NET_FILE, ROUTE_FILE, [VTYPES_FILE], **params)


class TestPlanDisruptionSpecs:
    def test_accident_slows_every_edge(self) -> None:
        specs = plan_disruption_specs("inc-1", "accident", {"a": 2, "b": 3}, 100, 600)

        assert [(s.type, s.edge_id, s.speed_factor) for s in specs] == [
            ("accident", "a", 0.25),
            ("accident", "b", 0.25),
        ]
        assert all(s.start_s == 100 and s.duration_s == 600 for s in specs)

    def test_closure_has_no_speed_factor(self) -> None:
        (spec,) = plan_disruption_specs("inc-1", "closure", {"a": 2}, 0, 60)

        assert spec.type == "closure"
        assert spec.speed_factor is None

    def test_weather_derates(self) -> None:
        (spec,) = plan_disruption_specs("inc-1", "weather", {"a": 2}, 0, 60)

        assert (spec.type, spec.speed_factor) == ("weather", 0.6)

    def test_construction_drops_a_lane_on_a_multi_lane_edge(self) -> None:
        (spec,) = plan_disruption_specs("inc-1", "construction", {"a": 3}, 0, 60)

        assert spec.type == "construction"
        assert spec.lane_indices == [0]

    def test_construction_on_a_single_lane_edge_slows_instead_of_closing_it(self) -> None:
        (spec,) = plan_disruption_specs("inc-1", "construction", {"a": 1}, 0, 60)

        assert spec.type == "accident"
        assert spec.speed_factor == 0.5

    def test_classes_without_a_road_effect_plan_nothing(self) -> None:
        assert plan_disruption_specs("inc-1", "sensor_fault", {"a": 2}, 0, 60) == []

    def test_spec_names_are_unique_per_edge(self) -> None:
        specs = plan_disruption_specs("inc-1", "accident", {"a": 2, "b": 2}, 0, 60)

        assert len({s.name for s in specs}) == 2


@requires_network
class TestGeometry:
    def test_geometry_is_real_lon_lat_for_every_road(self) -> None:
        service = _make_service()

        geometry = service.geometry()

        assert len(geometry) > 50
        for edge in geometry:
            coords: list[list[float]] = edge["coords"]  # type: ignore[assignment]
            assert len(coords) >= 2
            for lon, lat in coords:
                assert LON_RANGE[0] < lon < LON_RANGE[1]
                assert LAT_RANGE[0] < lat < LAT_RANGE[1]

    def test_snapshot_before_the_simulation_starts_is_not_ready(self) -> None:
        service = _make_service()

        assert service.snapshot()["ready"] is False


@requires_network
class TestIncidentResolution:
    def test_edge_id_resolves_to_that_road(self) -> None:
        service = _make_service()

        edges = service.apply_incident(
            {"incident_id": "i1", "class": "accident", "location": {"link_ids": ["1117867028"]}}
        )

        assert edges == ["1117867028"]

    def test_junction_id_resolves_to_its_incoming_roads(self) -> None:
        service = _make_service()

        edges = service.apply_incident(
            {"incident_id": "i1", "class": "accident", "location": {"link_ids": [JUNCTION]}}
        )

        assert set(edges) == JUNCTION_INCOMING

    def test_class_without_a_road_effect_is_ignored(self) -> None:
        service = _make_service()

        edges = service.apply_incident(
            {"incident_id": "i1", "class": "sensor_fault", "location": {"link_ids": [JUNCTION]}}
        )

        assert edges == []

    def test_unknown_location_is_ignored(self) -> None:
        service = _make_service()

        edges = service.apply_incident(
            {"incident_id": "i1", "class": "accident", "location": {"link_ids": ["nowhere"]}}
        )

        assert edges == []

    def test_resubmitting_the_same_incident_does_not_apply_it_twice(self) -> None:
        service = _make_service()
        incident = {"incident_id": "i1", "class": "closure", "location": {"link_ids": [JUNCTION]}}

        first = service.apply_incident(incident)
        second = service.apply_incident(incident)

        assert first
        assert second == []


@pytest.fixture
def running_service() -> Iterator[LiveTrafficService]:
    service = _make_service(disruption_duration_s=30)
    service.start_simulation()
    try:
        yield service
    finally:
        service._close_simulation()


@requires_network
class TestRealSimulation:
    def test_first_snapshot_has_real_vehicles_inside_midtown(
        self, running_service: LiveTrafficService
    ) -> None:
        snapshot = running_service.snapshot()

        assert snapshot["ready"] is True
        assert snapshot["sim_time_s"] == 60.0
        vehicles: list[list[float]] = snapshot["vehicles"]  # type: ignore[assignment]
        assert len(vehicles) > 0
        for lon, lat, speed in vehicles:
            assert LON_RANGE[0] < lon < LON_RANGE[1]
            assert LAT_RANGE[0] < lat < LAT_RANGE[1]
            assert speed >= 0

    def test_tick_advances_the_simulation_clock(self, running_service: LiveTrafficService) -> None:
        running_service.tick()

        assert running_service.snapshot()["sim_time_s"] == 65.0

    def test_per_road_stats_cover_every_road_with_sane_speed_ratios(
        self, running_service: LiveTrafficService
    ) -> None:
        edges: dict[str, dict[str, Any]] = running_service.snapshot()["edges"]  # type: ignore[assignment]

        assert set(edges) == {str(e["edge_id"]) for e in running_service.geometry()}
        ratios = [e["speed_ratio"] for e in edges.values() if e["speed_ratio"] is not None]
        assert ratios
        assert all(0.0 <= r <= 1.0 for r in ratios)

    def test_incident_becomes_a_listed_disruption_that_later_expires(
        self, running_service: LiveTrafficService
    ) -> None:
        running_service.apply_incident(
            {"incident_id": "inc-acc", "class": "accident", "location": {"link_ids": [JUNCTION]}}
        )

        running_service.tick()
        disruptions: list[dict[str, Any]] = running_service.snapshot()["disruptions"]  # type: ignore[assignment]

        assert {d["edge_id"] for d in disruptions} == JUNCTION_INCOMING
        assert {d["type"] for d in disruptions} == {"accident"}
        assert {d["incident_id"] for d in disruptions} == {"inc-acc"}

        for _ in range(8):  # 8 ticks x 5s = 40s of sim time, past the 30s duration
            running_service.tick()
        assert running_service.snapshot()["disruptions"] == []

    def test_closure_really_empties_the_closed_road(
        self, running_service: LiveTrafficService
    ) -> None:
        running_service.apply_incident(
            {
                "incident_id": "inc-closure",
                "class": "closure",
                "location": {"link_ids": ["1117867028"]},
            }
        )

        for _ in range(4):
            running_service.tick()
        edges: dict[str, dict[str, Any]] = running_service.snapshot()["edges"]  # type: ignore[assignment]

        assert edges["1117867028"]["vehicles"] == 0


@pytest.fixture
def control_service() -> Iterator[LiveTrafficService]:
    service = _make_service(disruption_duration_s=600, impact_window_s=20.0)
    service.start_simulation()
    try:
        yield service
    finally:
        service._close_simulation()


def _snapshot_of(service: LiveTrafficService, key: str) -> Any:
    return service.snapshot()[key]


@requires_network
class TestJunctionControl:
    def test_snapshot_reports_every_atlas_controlled_junction(
        self, control_service: LiveTrafficService
    ) -> None:
        junctions = _snapshot_of(control_service, "junctions")

        assert set(junctions) == {"42430333", "42430338", "42430342", "42430344", "42430347"}
        assert {j["mode"] for j in junctions.values()} == {"fixed_time"}
        assert all(j["queue"] >= 0 for j in junctions.values())

    def test_set_junction_mode_takes_effect_on_the_next_tick(
        self, control_service: LiveTrafficService
    ) -> None:
        control_service.set_junction_mode(JUNCTION, "actuated")
        control_service.tick()

        assert _snapshot_of(control_service, "junctions")[JUNCTION]["mode"] == "actuated"

    def test_the_chosen_mode_survives_a_simulation_restart(
        self, control_service: LiveTrafficService
    ) -> None:
        control_service.set_junction_mode(JUNCTION, "max_pressure")

        control_service.start_simulation()

        assert _snapshot_of(control_service, "junctions")[JUNCTION]["mode"] == "max_pressure"

    def test_unknown_junction_and_unavailable_mode_are_rejected(
        self, control_service: LiveTrafficService
    ) -> None:
        with pytest.raises(KeyError):
            control_service.set_junction_mode("42428283", "max_pressure")
        with pytest.raises(ValueError, match="mode must be one of"):
            control_service.set_junction_mode(JUNCTION, "rl")


@requires_network
class TestPlaybookApplicationAndImpact:
    def test_applied_max_pressure_playbook_overrides_the_junction_and_is_listed(
        self, control_service: LiveTrafficService
    ) -> None:
        queued = control_service.apply_playbook(
            "inc-mp", "switch_to_max_pressure", {"junction_ids": [JUNCTION]}
        )
        control_service.tick()

        junction = _snapshot_of(control_service, "junctions")[JUNCTION]
        playbooks = _snapshot_of(control_service, "playbooks")
        assert queued is True
        assert (junction["mode"], junction["playbook_override"]) == ("max_pressure", True)
        assert [p["incident_id"] for p in playbooks] == ["inc-mp"]

    def test_impact_is_measured_before_and_after_then_completes(
        self, control_service: LiveTrafficService
    ) -> None:
        control_service.apply_playbook(
            "inc-mp", "switch_to_max_pressure", {"junction_ids": [JUNCTION]}
        )
        control_service.tick()
        measuring = _snapshot_of(control_service, "impacts")["inc-mp"]
        assert measuring["status"] == "measuring"
        assert measuring["before_halted"] is not None

        for _ in range(6):  # 6 ticks x 5 s = 30 s > the 20 s window
            control_service.tick()
        done = _snapshot_of(control_service, "impacts")["inc-mp"]

        assert done["status"] == "complete"
        assert done["after_halted"] is not None
        assert done["playbook"] == "switch_to_max_pressure"
        assert done["edge_ids"]

    def test_divert_is_applied_to_the_roads_it_names(
        self, control_service: LiveTrafficService
    ) -> None:
        control_service.apply_playbook(
            "inc-div",
            "divert_arterial",
            {"from_edge": JUNCTION, "to_edge": JUNCTION, "duration_s": 600.0},
        )
        control_service.tick()

        impact = _snapshot_of(control_service, "impacts")["inc-div"]
        assert set(impact["edge_ids"]) == JUNCTION_INCOMING

    def test_playbook_the_twin_cannot_model_is_reported_not_applied(
        self, control_service: LiveTrafficService
    ) -> None:
        control_service.apply_playbook("inc-bus", "bus_bridge", {"route_id": "r1"})
        control_service.tick()

        impact = _snapshot_of(control_service, "impacts")["inc-bus"]
        assert impact["status"] == "not_applied"
        assert _snapshot_of(control_service, "playbooks") == []

    def test_hold_current_plan_changes_nothing(self, control_service: LiveTrafficService) -> None:
        control_service.apply_playbook("inc-hold", "hold_current_plan", {"reason": "r"})
        control_service.tick()

        assert _snapshot_of(control_service, "impacts")["inc-hold"]["status"] == "no_change"
        assert _snapshot_of(control_service, "playbooks") == []

    def test_a_response_is_applied_at_most_once_per_incident(
        self, control_service: LiveTrafficService
    ) -> None:
        first = control_service.apply_playbook("inc-1", "hold_current_plan", {"reason": "r"})
        second = control_service.apply_playbook("inc-1", "hold_current_plan", {"reason": "r"})

        assert (first, second) == (True, False)


@requires_network
class TestBackgroundThread:
    def test_thread_publishes_a_ready_snapshot_and_keeps_advancing(self) -> None:
        service = _make_service(warmup_s=30, tick_interval_s=0.05)
        service.start()
        try:
            assert service.wait_until_ready(60.0)
            first_time: float = service.snapshot()["sim_time_s"]  # type: ignore[assignment]
            for _ in range(200):
                if service.snapshot()["sim_time_s"] > first_time:  # type: ignore[operator]
                    break
                time.sleep(0.05)
            assert service.snapshot()["sim_time_s"] > first_time  # type: ignore[operator]
        finally:
            service.stop()
