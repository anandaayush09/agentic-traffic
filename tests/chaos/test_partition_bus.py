"""Chaos scenario 3/3 (docs/design/11_evaluation.md section 11.5.4): "partition the
bus for area X (fallback correctness)."

A real bug was found writing this scenario, not just re-narrating an existing test:
`IntersectionAgentService.decide_and_step()` had no exception handling around its bus
publish calls at all, so a bus_producer that EXISTS but starts failing (a real
partition -- distinct from `test_works_without_a_bus_producer`'s "never had a bus"
case) crashed the entire control loop, directly contradicting this project's own
stated "publishing is best-effort, not load-bearing for the decision itself"
principle. Fixed in `atlas.services.intersection_agent._publish_command`/
`_publish_heartbeat` (now catch and log, never propagate); the fix's own unit test
lives in tests/unit/test_intersection_agent_service.py::
TestIntersectionAgentApp::test_survives_a_bus_partition_mid_run, verified against a
real running SUMO twin. This module re-asserts the same real behavior under the
chaos-scenario framing docs/design/11_evaluation.md section 11.5.4 asks for.
"""

from __future__ import annotations

from atlas.bus.producer import BusProducer
from atlas.bus.schema_registry import SchemaRegistry
from atlas.common.config import project_root
from atlas.control.shield import ShieldConfig
from atlas.services.intersection_agent import IntersectionAgentConfig, IntersectionAgentService

NETWORK_DIR = project_root() / "data" / "networks" / "sandbox"


class _PartitionedRawProducer:
    """A bus connection that exists but cannot actually reach the broker -- every
    send() fails, as a real partitioned/unreachable Redpanda cluster eventually
    surfaces to a kafka-python producer."""

    def send(self, topic: str, value: bytes | None = None, key: bytes | None = None) -> object:
        raise ConnectionError("simulated partition: broker for this area is unreachable")

    def flush(self, timeout: float | None = None) -> None:
        pass


def test_scenario_bus_partitioned_mid_episode_control_loop_keeps_running() -> None:
    """Pass criterion: an intersection agent with a real (schema-validating)
    BusProducer wired to a partitioned connection must keep making real control
    decisions and stepping the real twin across MULTIPLE decision cycles during the
    partition -- fallback correctness means the agent's local decision loop is
    entirely independent of bus reachability, not merely that one call doesn't crash.
    """
    config = IntersectionAgentConfig(
        tls_id="B1",
        net_file=NETWORK_DIR / "net.net.xml",
        route_file=NETWORK_DIR / "routes.rou.xml",
        max_lanes=4,
        max_green_phases=2,
        decision_interval_s=5.0,
        shield_config=ShieldConfig(min_green_s=10.0, max_green_s=60.0),
        default_mode="max_pressure",
    )
    partitioned_bus = BusProducer(_PartitionedRawProducer(), SchemaRegistry())
    service = IntersectionAgentService(config, bus_producer=partitioned_bus)
    service.start()
    try:
        for _ in range(10):  # ten real decision cycles, all during the "partition"
            info = service.decide_and_step()
            assert "pressure" in info
    finally:
        service.stop()
