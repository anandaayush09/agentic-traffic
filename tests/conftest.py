"""Shared pytest fixtures.

`sandbox_twin_config` generates the Week 1 placeholder network exactly once per test
session (it is slow-ish -- two subprocess calls to SUMO tools -- and deterministic given
its fixed seed, so re-running it per test would only slow the suite down for no benefit).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from atlas.common.config import load_yaml_model, project_root  # noqa: E402
from atlas.twin import TwinEnvConfig  # noqa: E402


@pytest.fixture(scope="session")
def sandbox_twin_config() -> TwinEnvConfig:
    import generate_sandbox_network

    generate_sandbox_network.generate()

    root = project_root()
    config_path = root / "configs" / "network" / "sandbox.yaml"
    config = load_yaml_model(config_path, TwinEnvConfig)
    return config.model_copy(
        update={
            "net_file": root / config.net_file,
            "route_file": root / config.route_file,
        }
    )
