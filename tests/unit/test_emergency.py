"""Validates EmergencyVehicleInjector against a real running sandbox twin -- vehicle
injection actually landing in libsumo at the right sim-time, not a mock.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from atlas.common.config import project_root
from atlas.twin.emergency import EmergencyVehicleInjector, EmergencyVehicleSpec
from atlas.twin.env import SumoTwinEnv, TwinEnvConfig


@pytest.fixture(scope="session")
def sandbox_net_route() -> tuple[Path, Path]:
    import sys

    sys.path.insert(0, str(project_root() / "scripts"))
    import generate_sandbox_network

    generate_sandbox_network.generate()
    return generate_sandbox_network.NET_FILE, generate_sandbox_network.ROUTE_FILE


def _direct_edges(net_file: Path) -> list[str]:
    import sumolib

    net = sumolib.net.readNet(str(net_file))
    # A1 -> B1 is a real direct edge in the sandbox network (verified in
    # test_corridor_geometry.py's fallback-distance test).
    return [
        e.getID()
        for e in net.getEdges()
        if e.getFromNode().getID() == "A1" and e.getToNode().getID() == "B1"
    ]


class TestEmergencyVehicleInjector:
    def test_does_not_inject_before_depart_time(self, sandbox_net_route: tuple[Path, Path]) -> None:
        net_file, route_file = sandbox_net_route
        twin = SumoTwinEnv(
            TwinEnvConfig(name="emergency_test", net_file=net_file, route_file=route_file)
        )
        twin.reset()
        try:
            spec = EmergencyVehicleSpec(
                vehicle_id="emg_1",
                route_edges=_direct_edges(net_file),
                depart_s=100.0,
                vtype_id="DEFAULT_VEHTYPE",  # sandbox network has no emergency_vtype defined
            )
            injector = EmergencyVehicleInjector(twin, spec)

            injector.maybe_inject()

            assert injector.injected is False
            assert injector.is_active() is False
        finally:
            twin.close()

    def test_injects_once_depart_time_is_reached(
        self, sandbox_net_route: tuple[Path, Path]
    ) -> None:
        net_file, route_file = sandbox_net_route
        twin = SumoTwinEnv(
            TwinEnvConfig(name="emergency_test", net_file=net_file, route_file=route_file)
        )
        twin.reset()
        try:
            spec = EmergencyVehicleSpec(
                vehicle_id="emg_2",
                route_edges=_direct_edges(net_file),
                depart_s=0.0,
                vtype_id="DEFAULT_VEHTYPE",
            )
            injector = EmergencyVehicleInjector(twin, spec)

            injector.maybe_inject()
            twin.step()

            assert injector.injected is True
            assert injector.is_active() is True
            assert "emg_2" in twin.connection.vehicle.getIDList()
        finally:
            twin.close()

    def test_injects_exactly_once_even_if_called_repeatedly(
        self, sandbox_net_route: tuple[Path, Path]
    ) -> None:
        net_file, route_file = sandbox_net_route
        twin = SumoTwinEnv(
            TwinEnvConfig(name="emergency_test", net_file=net_file, route_file=route_file)
        )
        twin.reset()
        try:
            spec = EmergencyVehicleSpec(
                vehicle_id="emg_3",
                route_edges=_direct_edges(net_file),
                depart_s=0.0,
                vtype_id="DEFAULT_VEHTYPE",
            )
            injector = EmergencyVehicleInjector(twin, spec)

            injector.maybe_inject()
            injector.maybe_inject()  # must not raise "vehicle already exists"
            injector.maybe_inject()

            assert injector.injected is True
        finally:
            twin.close()

    def test_uses_the_emergency_vtype_by_default(
        self, sandbox_net_route: tuple[Path, Path]
    ) -> None:
        spec = EmergencyVehicleSpec(vehicle_id="emg_4", route_edges=["x"], depart_s=0.0)
        assert spec.vtype_id == "emergency_vtype"
