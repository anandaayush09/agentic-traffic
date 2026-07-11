#!/usr/bin/env python
"""Week 1 deliverable: run 1000 simulation steps on the sandbox digital twin and log
junction state snapshots along the way.

This is the executable proof that `atlas.twin.SumoTwinEnv` actually drives a SUMO
simulation end-to-end (network -> demand -> stepping -> signal/lane state readout), ahead
of anything being wired to the event bus (that starts in Week 5).

Usage:
    python scripts/run_twin_smoke_test.py [--steps 1000] [--log-every 100]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atlas.common.config import load_yaml_model, project_root  # noqa: E402
from atlas.common.logging import configure_logging, get_logger  # noqa: E402
from atlas.twin import SumoTwinEnv, TwinEnvConfig  # noqa: E402

configure_logging()
logger = get_logger(__name__)

CONFIG_PATH = project_root() / "configs" / "network" / "sandbox.yaml"


def load_config() -> TwinEnvConfig:
    config = load_yaml_model(CONFIG_PATH, TwinEnvConfig)
    root = project_root()
    # YAML stores paths relative to the repo root; resolve them here so the config
    # itself stays portable and doesn't hardcode an absolute machine path.
    return config.model_copy(
        update={
            "net_file": root / config.net_file,
            "route_file": root / config.route_file,
        }
    )


def ensure_network_exists(config: TwinEnvConfig) -> None:
    if config.net_file.is_file() and config.route_file.is_file():
        return
    logger.info("smoke_test.generating_missing_network")
    import generate_sandbox_network

    generate_sandbox_network.generate()


def run(steps: int, log_every: int) -> None:
    config = load_config()
    ensure_network_exists(config)

    with SumoTwinEnv(config) as env:
        logger.info(
            "smoke_test.started",
            twin_name=config.name,
            traffic_lights=env.get_traffic_light_ids(),
            steps=steps,
        )
        for step in range(1, steps + 1):
            env.step()
            if step % log_every == 0:
                states = env.get_all_junction_states()
                for tls_id, state in states.items():
                    total_queue = sum(lane.queue_length for lane in state.incoming_lanes)
                    logger.info(
                        "smoke_test.junction_state",
                        step=step,
                        sim_time=env.sim_time,
                        tls_id=tls_id,
                        phase=state.phase_index,
                        phase_state=state.phase_state,
                        total_queue=total_queue,
                        incoming_lanes=len(state.incoming_lanes),
                    )
        logger.info("smoke_test.completed", steps=steps, final_sim_time=env.sim_time)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()
    run(steps=args.steps, log_every=args.log_every)


if __name__ == "__main__":
    main()
