"""Fleet mix: vehicle-type diversity (passenger/truck/bus/emergency) layered on top of
time-of-day demand (docs/design/08_data_strategy.md section 8.2). Real GTFS-derived bus
routes are explicitly out of scope here -- see configs/fleet/manhattan_midtown.yaml.

Mechanism (verified against randomTrips.py directly before implementing): `--vehicle-
class` and a `type="..."` trip-attribute cannot be combined (randomTrips.py itself
rejects it) -- so each vehicle type is defined once as a SUMO `<vType>` (carrying its own
`vClass`) in a shared additional file, and randomTrips.py is invoked once per fleet
component with `--trip-attributes 'type="<vtype_id>"'` referencing it. Each component's
`randomTrips.py -p` (mean seconds between insertions) is scaled by `period_s / fraction`
so that mixing all components back together reproduces the period's overall target
volume, split according to the configured fractions.
"""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree

from pydantic import BaseModel, Field, model_validator

FRACTION_SUM_TOLERANCE = 1e-6


class VehicleTypeSpec(BaseModel):
    """A SUMO <vType>: physical/behavioral parameters for one vehicle class."""

    vtype_id: str
    vclass: str
    length_m: float = Field(gt=0)
    max_speed_mps: float = Field(gt=0)
    accel_mps2: float = Field(gt=0)
    decel_mps2: float = Field(gt=0)
    sigma: float = Field(ge=0, le=1)
    """Driver imperfection, SUMO's Krauss model parameter (0=perfect, 1=very imperfect)."""


class FleetComponent(BaseModel):
    """What fraction of total period volume should be generated as a given vehicle type."""

    vtype_id: str
    fraction: float = Field(gt=0, le=1)


class FleetConfig(BaseModel):
    network: str
    vehicle_types: list[VehicleTypeSpec]
    components: list[FleetComponent]

    @model_validator(mode="after")
    def _validate_components(self) -> FleetConfig:
        known_ids = {vt.vtype_id for vt in self.vehicle_types}
        for component in self.components:
            if component.vtype_id not in known_ids:
                raise ValueError(f"component references unknown vtype_id '{component.vtype_id}'")
        total = sum(c.fraction for c in self.components)
        if abs(total - 1.0) > FRACTION_SUM_TOLERANCE:
            raise ValueError(f"component fractions must sum to 1.0, got {total}")
        return self


def write_vtypes_additional_file(fleet_config: FleetConfig, out_path: Path) -> Path:
    """Writes the <additional><vType .../></additional> file that TwinEnvConfig.
    additional_files points SUMO at, so every generated <vehicle type="..."/> resolves.
    """
    root = ElementTree.Element("additional")
    for vt in fleet_config.vehicle_types:
        ElementTree.SubElement(
            root,
            "vType",
            {
                "id": vt.vtype_id,
                "vClass": vt.vclass,
                "length": f"{vt.length_m:.2f}",
                "maxSpeed": f"{vt.max_speed_mps:.2f}",
                "accel": f"{vt.accel_mps2:.2f}",
                "decel": f"{vt.decel_mps2:.2f}",
                "sigma": f"{vt.sigma:.2f}",
            },
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ElementTree.indent(root, space="  ")
    ElementTree.ElementTree(root).write(out_path, encoding="UTF-8", xml_declaration=True)
    return out_path


def merge_route_files(route_files: list[Path], out_path: Path) -> Path:
    """Merges the <vehicle> elements (each with its nested <route>) from multiple
    per-vehicle-class route files into one, sorted by depart time ascending -- the order
    SUMO requires. Vehicle ids are assumed already unique across inputs (each fleet
    component's randomTrips.py run uses a distinct --prefix).
    """
    vehicles: list[tuple[float, ElementTree.Element]] = []
    for route_file in route_files:
        tree = ElementTree.parse(route_file)
        for vehicle in tree.getroot().findall("vehicle"):
            depart = float(vehicle.get("depart", "0"))
            vehicles.append((depart, vehicle))

    vehicles.sort(key=lambda item: item[0])

    root = ElementTree.Element("routes")
    for _, vehicle in vehicles:
        root.append(vehicle)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ElementTree.indent(root, space="  ")
    ElementTree.ElementTree(root).write(out_path, encoding="UTF-8", xml_declaration=True)
    return out_path
