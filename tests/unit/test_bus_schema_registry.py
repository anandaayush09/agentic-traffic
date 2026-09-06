"""Validates SchemaRegistry against the two committed Week 5 schemas
(schemas/sensor.detections.v1.schema.json, schemas/control.commands.v1.schema.json) --
both that they are themselves well-formed JSON Schema documents (the "CI schema-
validation" the Week 5 roadmap asks for) and that they correctly accept/reject real
payload shapes.
"""

from __future__ import annotations

import jsonschema
import pytest

from atlas.bus.schema_registry import SchemaRegistry, SchemaValidationError

VALID_SENSOR_DETECTIONS = {
    "msg_id": "5b1b2b2e-8f1a-4b1a-9b1a-2b1a2b1a2b1a",
    "ts": "2026-09-06T04:00:00+00:00",
    "schema_version": "1",
    "intersection_id": "42430333",
    "lanes": [
        {
            "lane_id": "-123456_0",
            "queue_length": 3,
            "occupancy": 0.42,
            "mean_speed_mps": 4.5,
            "vehicle_count": 5,
        }
    ],
}

VALID_CONTROL_COMMAND = {
    "msg_id": "5b1b2b2e-8f1a-4b1a-9b1a-2b1a2b1a2b1a",
    "ts": "2026-09-06T04:00:00+00:00",
    "schema_version": "1",
    "intersection_id": "42430333",
    "actor": "rl_agent",
    "action": {"kind": "set_phase", "phase_id": 0},
    "policy_version": "mappo_corridor_v1",
}


class TestSchemasAreWellFormed:
    """The "CI schema-validation" the roadmap asks for: every committed schema must
    itself be a valid JSON Schema document, checked in the same suite CI already runs."""

    def test_sensor_detections_schema_is_well_formed(self) -> None:
        registry = SchemaRegistry()
        jsonschema.Draft202012Validator.check_schema(registry.schema_for("sensor.detections.v1"))

    def test_control_commands_schema_is_well_formed(self) -> None:
        registry = SchemaRegistry()
        jsonschema.Draft202012Validator.check_schema(registry.schema_for("control.commands.v1"))


class TestUnknownTopic:
    def test_raises_clear_error(self) -> None:
        registry = SchemaRegistry()
        with pytest.raises(SchemaValidationError, match="No committed schema"):
            registry.schema_for("nonexistent.topic.v1")


class TestSensorDetectionsValidation:
    def test_accepts_valid_payload(self) -> None:
        SchemaRegistry().validate("sensor.detections.v1", VALID_SENSOR_DETECTIONS)

    def test_rejects_missing_intersection_id(self) -> None:
        payload = {k: v for k, v in VALID_SENSOR_DETECTIONS.items() if k != "intersection_id"}
        with pytest.raises(SchemaValidationError):
            SchemaRegistry().validate("sensor.detections.v1", payload)

    def test_rejects_empty_lanes(self) -> None:
        payload = {**VALID_SENSOR_DETECTIONS, "lanes": []}
        with pytest.raises(SchemaValidationError):
            SchemaRegistry().validate("sensor.detections.v1", payload)

    def test_rejects_occupancy_out_of_range(self) -> None:
        payload = {
            **VALID_SENSOR_DETECTIONS,
            "lanes": [{**VALID_SENSOR_DETECTIONS["lanes"][0], "occupancy": 1.5}],
        }
        with pytest.raises(SchemaValidationError):
            SchemaRegistry().validate("sensor.detections.v1", payload)

    def test_rejects_unknown_field(self) -> None:
        payload = {**VALID_SENSOR_DETECTIONS, "unexpected_field": "nope"}
        with pytest.raises(SchemaValidationError):
            SchemaRegistry().validate("sensor.detections.v1", payload)


class TestControlCommandsValidation:
    def test_accepts_valid_payload(self) -> None:
        SchemaRegistry().validate("control.commands.v1", VALID_CONTROL_COMMAND)

    def test_rejects_unknown_actor(self) -> None:
        payload = {**VALID_CONTROL_COMMAND, "actor": "not_a_real_actor"}
        with pytest.raises(SchemaValidationError):
            SchemaRegistry().validate("control.commands.v1", payload)

    def test_rejects_missing_action_kind(self) -> None:
        payload = {**VALID_CONTROL_COMMAND, "action": {"phase_id": 0}}
        with pytest.raises(SchemaValidationError):
            SchemaRegistry().validate("control.commands.v1", payload)

    def test_accepts_real_manhattan_tls_id_not_matching_design_docs_illustrative_pattern(
        self,
    ) -> None:
        """The design doc's representative schema used '^J[0-9]{3}$'; real tls_ids
        (Week 2) look nothing like that -- e.g. 'GS_42439981' (see AUDIT.md's grouped-
        signal caveat). This schema is intentionally adapted, not a literal copy."""
        payload = {**VALID_CONTROL_COMMAND, "intersection_id": "GS_42439981"}
        SchemaRegistry().validate("control.commands.v1", payload)
