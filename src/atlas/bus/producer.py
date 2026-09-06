"""BusProducer: schema validation and envelope generation before anything reaches the
wire (docs/design/05_communication.md principle 4, "schema or it didn't happen"). Wraps a
raw producer via structural typing (`Protocol`) rather than importing kafka-python's
`KafkaProducer` into this module's type surface directly -- `BusProducer(raw, registry)`
works with any object matching `_RawProducer`'s send/flush shape, a real kafka-python
instance or a test double, with no adapter class needed (kafka-python's own
`KafkaProducer.send(topic, value=..., key=...)` signature already matches).
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any, Protocol

from atlas.bus.schema_registry import SchemaRegistry


class _RawProducer(Protocol):
    def send(self, topic: str, value: bytes | None = None, key: bytes | None = None) -> object: ...
    def flush(self, timeout: float | None = None) -> None: ...


class BusProducer:
    def __init__(self, raw: _RawProducer, registry: SchemaRegistry | None = None) -> None:
        self._raw = raw
        self._registry = registry or SchemaRegistry()

    @classmethod
    def connect(cls, brokers: list[str]) -> BusProducer:
        from kafka import KafkaProducer

        return cls(KafkaProducer(bootstrap_servers=brokers))

    def publish(
        self, topic: str, payload: dict[str, Any], key: str | None = None
    ) -> dict[str, Any]:
        """Fills envelope defaults (`msg_id`, `ts`, `schema_version`) if the caller
        didn't supply them, validates the result against the topic's committed schema,
        then sends. Returns the exact message sent (including generated envelope
        fields) so callers/tests can inspect what actually went on the wire. Raises
        `SchemaValidationError` -- and sends nothing -- if the message doesn't conform.
        """
        message = self._with_envelope_defaults(payload)
        self._registry.validate(topic, message)
        self._raw.send(
            topic,
            value=json.dumps(message).encode("utf-8"),
            key=key.encode("utf-8") if key is not None else None,
        )
        return message

    def publish_raw(self, topic: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Publishes without schema validation. Used only for DLQ topics, where the
        entire reason the message exists is that it failed its original topic's schema
        -- never use this for a real data topic.
        """
        message = self._with_envelope_defaults(payload)
        self._raw.send(topic, value=json.dumps(message).encode("utf-8"))
        return message

    def flush(self) -> None:
        self._raw.flush()

    @staticmethod
    def _with_envelope_defaults(payload: dict[str, Any]) -> dict[str, Any]:
        message = dict(payload)
        message.setdefault("msg_id", str(uuid.uuid4()))
        message.setdefault("ts", datetime.now(UTC).isoformat())
        message.setdefault("schema_version", "1")
        return message
