"""Reinforcement learning: the Week 3 single-junction Gymnasium environment and (later)
the from-scratch multi-agent training code. Deliberately a separate package from
`atlas.control` -- this is where a *policy* is trained or executed, `atlas.control` is
where the safety envelope and classical controllers it must respect live.

`RolloutWorker` (atlas.rl.rollout) is deliberately NOT re-exported here, unlike
everything else in this package. `import ray` alone -- verified, not even `ray.init()` is
needed -- permanently breaks subsequent `import libsumo` in the same process (see
rollout.py's module docstring for the full diagnosis). Since nearly every other test file
does `from atlas.rl import ...`, eagerly importing rollout.py here would make `ray` a
transitive import of the whole package, silently poisoning every libsumo-based test that
happens to run afterward in the same pytest session -- exactly the regression this
comment exists to prevent from being reintroduced. Import it explicitly and only where
actually needed: `from atlas.rl.rollout import RolloutWorker`.
"""

from atlas.rl.model_io import patch_sb3_zip_loading
from atlas.rl.multi_agent_env import MultiAgentSignalEnv, MultiAgentSignalEnvConfig
from atlas.rl.single_junction_env import SingleJunctionEnv, SingleJunctionEnvConfig

__all__ = [
    "SingleJunctionEnv",
    "SingleJunctionEnvConfig",
    "patch_sb3_zip_loading",
    "MultiAgentSignalEnv",
    "MultiAgentSignalEnvConfig",
]
