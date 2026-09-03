"""Control plane: the fast loop (5-second signal decisions), shared safety envelope, and
classical controllers, per docs/design/03_system_architecture.md and 06_ai_pipeline.md.

Public API re-exported here so callers write `from atlas.control import SignalShield`
instead of reaching into submodules directly.
"""

from atlas.control.shield import ShieldConfig, ShieldDecision, SignalProgram, SignalShield

__all__ = [
    "SignalShield",
    "ShieldConfig",
    "ShieldDecision",
    "SignalProgram",
]
