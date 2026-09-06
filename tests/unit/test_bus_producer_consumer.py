"""Validates BusProducer/BusConsumer logic (envelope generation, schema validation,
DLQ routing, idempotent dedup) against fake raw producer/consumer doubles -- no real
broker needed, since BusProducer/BusConsumer depend on `_RawProducer`/`_RawRecord` only
structurally (Protocol), not on kafka-python directly. The real-broker round trip is
tests/integration/test_bus_roundtrip.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

from atlas.bus.consumer import BusConsumer
from atlas.bus.dedup import InMemoryDeduplicator
from atlas.bus.producer import BusProducer
from atlas.bus.schema_registry import SchemaRegistry, SchemaValidationError

VALID_SENSOR_DETECTIONS = {
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


@dataclass
class _FakeRawProducer:
    sent: list[tuple[str, bytes | None, bytes | None]] = field(default_factory=list)

    def send(self, topic: str, value: bytes | None = None, key: bytes | None = None) -> object:
        self.sent.append((topic, value, key))
        return None

    def flush(self, timeout: float | None = None) -> None:
        pass


@dataclass
class _FakeRecord:
    value: bytes


class TestBusProducerPublish:
    def test_fills_envelope_defaults_and_sends(self) -> None:
        raw = _FakeRawProducer()
        producer = BusProducer(raw)

        sent_message = producer.publish("sensor.detections.v1", VALID_SENSOR_DETECTIONS)

        assert "msg_id" in sent_message
        assert "ts" in sent_message
        assert sent_message["schema_version"] == "1"
        assert len(raw.sent) == 1
        topic, value, _ = raw.sent[0]
        assert topic == "sensor.detections.v1"
        assert json.loads(value) == sent_message  # type: ignore[arg-type]

    def test_does_not_override_caller_supplied_envelope_fields(self) -> None:
        raw = _FakeRawProducer()
        producer = BusProducer(raw)
        payload = {**VALID_SENSOR_DETECTIONS, "msg_id": "fixed-id", "schema_version": "1"}

        sent_message = producer.publish("sensor.detections.v1", payload)

        assert sent_message["msg_id"] == "fixed-id"

    def test_invalid_payload_raises_and_sends_nothing(self) -> None:
        raw = _FakeRawProducer()
        producer = BusProducer(raw)
        invalid_payload = {**VALID_SENSOR_DETECTIONS, "lanes": []}

        with pytest.raises(SchemaValidationError):
            producer.publish("sensor.detections.v1", invalid_payload)

        assert raw.sent == []


class TestBusConsumerConsume:
    def _valid_record(self) -> _FakeRecord:
        raw = _FakeRawProducer()
        BusProducer(raw).publish("sensor.detections.v1", VALID_SENSOR_DETECTIONS)
        _, value, _ = raw.sent[0]
        assert value is not None
        return _FakeRecord(value=value)

    def test_yields_valid_payload(self) -> None:
        record = self._valid_record()
        consumer = BusConsumer([record], topic="sensor.detections.v1")

        received = list(consumer.consume())

        assert len(received) == 1
        assert received[0]["intersection_id"] == "42430333"

    def test_routes_invalid_json_to_dlq(self) -> None:
        dlq_raw = _FakeRawProducer()
        dlq_producer = BusProducer(dlq_raw)
        bad_record = _FakeRecord(value=b"not valid json{{{")
        consumer = BusConsumer(
            [bad_record], topic="sensor.detections.v1", dlq_producer=dlq_producer
        )

        received = list(consumer.consume())

        assert received == []
        assert len(dlq_raw.sent) == 1
        dlq_topic, dlq_value, _ = dlq_raw.sent[0]
        assert dlq_topic == "sensor.detections.v1.dlq"
        assert json.loads(dlq_value)["reason"] == "invalid_json"  # type: ignore[arg-type]

    def test_routes_schema_invalid_payload_to_dlq(self) -> None:
        dlq_raw = _FakeRawProducer()
        dlq_producer = BusProducer(dlq_raw)
        invalid_message = {**VALID_SENSOR_DETECTIONS, "lanes": []}  # violates minItems
        bad_record = _FakeRecord(value=json.dumps(invalid_message).encode("utf-8"))
        consumer = BusConsumer(
            [bad_record], topic="sensor.detections.v1", dlq_producer=dlq_producer
        )

        received = list(consumer.consume())

        assert received == []
        assert len(dlq_raw.sent) == 1
        assert json.loads(dlq_raw.sent[0][1])["reason"] == "schema_invalid"  # type: ignore[index]

    def test_without_dlq_producer_silently_drops_invalid_messages(self) -> None:
        bad_record = _FakeRecord(value=b"not valid json{{{")
        consumer = BusConsumer([bad_record], topic="sensor.detections.v1")

        received = list(consumer.consume())

        assert received == []  # must not raise, per the "one poison message" contract

    def test_deduplicates_repeated_msg_id(self) -> None:
        record = self._valid_record()
        deduplicator = InMemoryDeduplicator()
        consumer = BusConsumer(
            [record, record], topic="sensor.detections.v1", deduplicator=deduplicator
        )

        received = list(consumer.consume())

        assert len(received) == 1  # second copy deduped, not yielded twice

    def test_round_trip_through_shared_registry_matches_exactly(self) -> None:
        """Produce via BusProducer, consume via BusConsumer, over a shared in-memory
        'broker' (a plain list) -- the same schema-validate-produce-consume path a real
        Redpanda round trip exercises, minus the actual network/broker."""
        registry = SchemaRegistry()
        raw_producer = _FakeRawProducer()
        producer = BusProducer(raw_producer, registry)

        sent = producer.publish("sensor.detections.v1", VALID_SENSOR_DETECTIONS)

        records = [_FakeRecord(value=value) for _, value, _ in raw_producer.sent]
        consumer = BusConsumer(records, topic="sensor.detections.v1", registry=registry)
        received = list(consumer.consume())

        assert received == [sent]
