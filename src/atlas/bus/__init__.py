"""Event bus: producer/consumer wrappers, schema validation, and idempotency for
Redpanda (Kafka API) -- docs/design/05_communication.md's "bus-only integration"
principle. Week 5's skeleton covers `sensor.detections.v1` and `control.commands.v1`
only: the first two of section 5.2's full topic taxonomy, enough to prove the schema-
validate-produce-consume-DLQ loop end to end before later weeks add the rest.
"""

from atlas.bus.consumer import BusConsumer
from atlas.bus.dedup import InMemoryDeduplicator, MessageDeduplicator
from atlas.bus.producer import BusProducer
from atlas.bus.schema_registry import SchemaRegistry, SchemaValidationError

__all__ = [
    "BusProducer",
    "BusConsumer",
    "SchemaRegistry",
    "SchemaValidationError",
    "MessageDeduplicator",
    "InMemoryDeduplicator",
]
