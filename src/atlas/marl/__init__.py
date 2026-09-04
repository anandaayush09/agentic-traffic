"""From-scratch MAPPO: the actual multi-agent training implementation (Week 5+), built on
top of atlas.rl's environments and atlas.control's safety envelope. Per the frozen
architecture, this is NOT RLlib -- see network.py's module docstring for why.
"""

from atlas.marl.buffer import RolloutBuffer
from atlas.marl.network import ActorCritic, build_action_mask
from atlas.marl.ppo import PPOConfig, PPOTrainer, PPOUpdateStats

__all__ = [
    "ActorCritic",
    "build_action_mask",
    "RolloutBuffer",
    "PPOConfig",
    "PPOTrainer",
    "PPOUpdateStats",
]
