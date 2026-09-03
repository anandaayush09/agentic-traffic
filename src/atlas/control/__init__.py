"""Control plane: the fast loop (5-second signal decisions), shared safety envelope, and
classical controllers, per docs/design/03_system_architecture.md and 06_ai_pipeline.md.

Public API re-exported here so callers write `from atlas.control import SignalShield`
instead of reaching into submodules directly.
"""

from atlas.control.fixed_time import (
    WebsterResult,
    compute_webster_green_splits,
    select_fixed_time_phase,
)
from atlas.control.max_pressure import (
    MaxPressureProgram,
    Movement,
    phase_pressure,
    select_max_pressure_phase,
)
from atlas.control.shield import ShieldConfig, ShieldDecision, SignalProgram, SignalShield

__all__ = [
    "SignalShield",
    "ShieldConfig",
    "ShieldDecision",
    "SignalProgram",
    "MaxPressureProgram",
    "Movement",
    "phase_pressure",
    "select_max_pressure_phase",
    "WebsterResult",
    "compute_webster_green_splits",
    "select_fixed_time_phase",
]
