"""BusConsumer: schema validation, idempotent dedup, and DLQ routing on the consume side
(docs/design/05_communication.md sections 5.1/5.6) -- the other half of "schema or it
didn't happen": a producer validating its own output is necessary but not sufficient,
since nothing stops a different producer, a bug, or a version skew from putting a
malformed message on the same topic.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from typing import Any, Protocol

from atlas.bus.dedup import InMemoryDeduplicator, MessageDeduplicator
from atlas.bus.producer import BusProducer
from atlas.bus.schema_registry import SchemaRegistry, SchemaValidationError


class _RawRecord(Protocol):
    value: bytes


class BusConsumer:
    def __init__(
        self,
        raw: Iterable[_RawRecord],
        topic: str,
        registry: SchemaRegistry | None = None,
        dlq_producer: BusProducer | None = None,
        deduplicator: MessageDeduplicator | None = None,
    ) -> None:
        self._raw = raw
        self._topic = topic
        self._registry = registry or SchemaRegistry()
        self._dlq_producer = dlq_producer
        self._deduplicator = deduplicator or InMemoryDeduplicator()

    @classmethod
    def connect(cls, brokers: list[str], topic: str, group_id: str) -> BusConsumer:
        from kafka import KafkaConsumer

        raw = KafkaConsumer(topic, bootstrap_servers=brokers, group_id=group_id)
        return cls(raw, topic, dlq_producer=BusProducer.connect(brokers))

    def consume(self) -> Iterator[dict[str, Any]]:
        """Yields validated, de-duplicated payloads only. Invalid JSON or schema
        failures are routed to `<topic>.dlq` (if a `dlq_producer` was supplied) and
        skipped, never raised -- one poison message must not stop the consumer loop
        (docs/design/05_communication.md section 5.6's "poison messages -> DLQ").
        """
        for record in self._raw:
            try:
                payload = json.loads(record.value)
            except json.JSONDecodeError:
                self._to_dlq(record.value, reason="invalid_json")
                continue

            msg_id = payload.get("msg_id")
            if msg_id is not None and self._deduplicator.seen(msg_id):
                continue

            try:
                self._registry.validate(self._topic, payload)
            except SchemaValidationError:
                self._to_dlq(record.value, reason="schema_invalid")
                continue

            if msg_id is not None:
                self._deduplicator.mark(msg_id)
            yield payload

    def _to_dlq(self, raw_value: bytes, reason: str) -> None:
        if self._dlq_producer is None:
            return
        self._dlq_producer.publish_raw(
            f"{self._topic}.dlq",
            {"raw_value_hex": raw_value.hex(), "reason": reason},
        )
