"""Validates fleet-mix config, vType file generation, route-file merging, and (against a
real generated period) that the resulting demand actually contains a plausible mix of all
configured vehicle types.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from xml.etree import ElementTree

import pytest
from pydantic import ValidationError

from atlas.common.config import load_yaml_model, project_root
from atlas.twin.fleet import (
    FleetComponent,
    FleetConfig,
    VehicleTypeSpec,
    merge_route_files,
    write_vtypes_additional_file,
)

FLEET_CONFIG_PATH = project_root() / "configs" / "fleet" / "manhattan_midtown.yaml"

_PASSENGER = VehicleTypeSpec(
    vtype_id="passenger_vtype",
    vclass="passenger",
    length_m=4.5,
    max_speed_mps=13.9,
    accel_mps2=2.6,
    decel_mps2=4.5,
    sigma=0.5,
)
_TRUCK = VehicleTypeSpec(
    vtype_id="truck_vtype",
    vclass="truck",
    length_m=8.0,
    max_speed_mps=11.1,
    accel_mps2=1.0,
    decel_mps2=3.5,
    sigma=0.5,
)


class TestFleetConfigValidation:
    def test_valid_config(self) -> None:
        config = FleetConfig(
            network="x",
            vehicle_types=[_PASSENGER, _TRUCK],
            components=[
                FleetComponent(vtype_id="passenger_vtype", fraction=0.9),
                FleetComponent(vtype_id="truck_vtype", fraction=0.1),
            ],
        )

        assert len(config.components) == 2

    def test_rejects_fractions_not_summing_to_one(self) -> None:
        with pytest.raises(ValidationError, match="must sum to 1.0"):
            FleetConfig(
                network="x",
                vehicle_types=[_PASSENGER, _TRUCK],
                components=[
                    FleetComponent(vtype_id="passenger_vtype", fraction=0.9),
                    FleetComponent(vtype_id="truck_vtype", fraction=0.2),
                ],
            )

    def test_rejects_unknown_vtype_reference(self) -> None:
        with pytest.raises(ValidationError, match="unknown vtype_id"):
            FleetConfig(
                network="x",
                vehicle_types=[_PASSENGER],
                components=[FleetComponent(vtype_id="not_defined", fraction=1.0)],
            )

    def test_loads_real_config(self) -> None:
        config = load_yaml_model(FLEET_CONFIG_PATH, FleetConfig)

        vtype_ids = {vt.vtype_id for vt in config.vehicle_types}
        assert vtype_ids == {
            "passenger_vtype",
            "truck_vtype",
            "bus_vtype",
            "emergency_vtype",
        }
        assert sum(c.fraction for c in config.components) == pytest.approx(1.0)


class TestWriteVtypesAdditionalFile:
    def test_writes_expected_vtype_elements(self, tmp_path: Path) -> None:
        config = FleetConfig(
            network="x",
            vehicle_types=[_PASSENGER, _TRUCK],
            components=[
                FleetComponent(vtype_id="passenger_vtype", fraction=0.9),
                FleetComponent(vtype_id="truck_vtype", fraction=0.1),
            ],
        )
        out_path = tmp_path / "vtypes.add.xml"

        write_vtypes_additional_file(config, out_path)

        tree = ElementTree.parse(out_path)
        vtypes = {vt.get("id"): vt for vt in tree.getroot().findall("vType")}
        assert set(vtypes.keys()) == {"passenger_vtype", "truck_vtype"}
        assert vtypes["truck_vtype"].get("vClass") == "truck"
        assert float(vtypes["truck_vtype"].get("length", "0")) == pytest.approx(8.0)


class TestMergeRouteFiles:
    def test_merges_and_sorts_by_depart_time(self, tmp_path: Path) -> None:
        file_a = tmp_path / "a.rou.xml"
        file_a.write_text(
            '<routes><vehicle id="a0" type="x" depart="5.00">'
            '<route edges="e1 e2"/></vehicle></routes>',
            encoding="utf-8",
        )
        file_b = tmp_path / "b.rou.xml"
        file_b.write_text(
            '<routes><vehicle id="b0" type="y" depart="1.00">'
            '<route edges="e3 e4"/></vehicle>'
            '<vehicle id="b1" type="y" depart="9.00">'
            '<route edges="e5 e6"/></vehicle></routes>',
            encoding="utf-8",
        )
        out_path = tmp_path / "merged.rou.xml"

        merge_route_files([file_a, file_b], out_path)

        tree = ElementTree.parse(out_path)
        vehicles = tree.getroot().findall("vehicle")
        ids_in_order = [v.get("id") for v in vehicles]
        assert ids_in_order == ["b0", "a0", "b1"]  # sorted 1.00, 5.00, 9.00


@pytest.fixture(scope="session")
def generated_fleet_demand() -> Path:
    import sys

    sys.path.insert(0, str(project_root() / "scripts"))
    import build_district_network
    import generate_demand

    build_district_network.build_network()
    generated = generate_demand.generate_all()
    return generated["pm_peak"]


class TestFleetMixIntegration:
    def test_merged_period_contains_all_configured_types(
        self, generated_fleet_demand: Path
    ) -> None:
        tree = ElementTree.parse(generated_fleet_demand)
        type_counts = Counter(v.get("type") for v in tree.getroot().findall("vehicle"))

        assert set(type_counts.keys()) == {
            "passenger_vtype",
            "truck_vtype",
            "bus_vtype",
            "emergency_vtype",
        }
        # Passenger must dominate (configured fraction 0.85) and emergency must be
        # rarest (0.02) -- a loose ordering check, not an exact-fraction match, since
        # randomTrips.py's insertion process is stochastic per component.
        assert type_counts["passenger_vtype"] > type_counts["truck_vtype"]
        assert type_counts["passenger_vtype"] > type_counts["bus_vtype"]
        assert type_counts["truck_vtype"] > type_counts["emergency_vtype"]
        assert type_counts["bus_vtype"] > type_counts["emergency_vtype"]

    def test_merged_period_is_depart_time_sorted(self, generated_fleet_demand: Path) -> None:
        tree = ElementTree.parse(generated_fleet_demand)
        departs = [float(v.get("depart", "0")) for v in tree.getroot().findall("vehicle")]

        assert departs == sorted(departs)
