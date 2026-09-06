"""Real Redpanda/Kafka round-trip: produce a message, consume it back, over an actual
running broker -- the Week 5 roadmap's "bus round-trip integration test". Requires a
broker reachable at KAFKA_BROKERS (default localhost:9092, i.e. `docker compose up
redpanda`); skips automatically if unreachable, since Docker is not installed on this dev
machine (see the ATLAS dev-environment memory) and this is expected to actually run in
CI/on a machine with Docker, not local dev -- the same honest verification split already
used for docker-compose.yml since Week 1: correct code, verified wherever verification is
actually possible, documented (not faked) where it isn't yet.
"""

from __future__ import annotations

import os
import socket
import uuid

import pytest

pytestmark = pytest.mark.integration

_KAFKA_BROKER = os.environ.get("KAFKA_BROKERS", "localhost:9092")


def _broker_reachable() -> bool:
    host, _, port_str = _KAFKA_BROKER.partition(":")
    try:
        with socket.create_connection((host, int(port_str or "9092")), timeout=1.0):
            return True
    except OSError:
        return False


@pytest.mark.skipif(
    not _broker_reachable(),
    reason=f"No Kafka/Redpanda broker reachable at KAFKA_BROKERS={_KAFKA_BROKER}",
)
def test_sensor_detections_round_trip() -> None:
    from atlas.bus import BusConsumer, BusProducer

    topic = "sensor.detections.v1"
    producer = BusProducer.connect([_KAFKA_BROKER])
    consumer = BusConsumer.connect([_KAFKA_BROKER], topic, group_id=f"test-{uuid.uuid4()}")

    payload = {
        "intersection_id": "42430333",
        "lanes": [
            {
                "lane_id": "-123456_0",
                "queue_length": 2,
                "occupancy": 0.3,
                "mean_speed_mps": 5.0,
                "vehicle_count": 3,
            }
        ],
    }
    sent = producer.publish(topic, payload)
    producer.flush()

    received = next(iter(consumer.consume()))
    assert received == sent
