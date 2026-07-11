"""Locates the SUMO installation bundled by the `eclipse-sumo` PyPI package.

We deliberately do not require the user to install SUMO separately or set `SUMO_HOME`
by hand: `eclipse-sumo` ships the full SUMO binary distribution (bin/, tools/, data/)
inside an importable `sumo` package, so its location can be resolved in pure Python and
is portable across the Windows/Linux/macOS wheels published for it. This keeps the
"one command bring-up" property from the repository design (Part 9) -- `pip install`
(or `uv sync`) is sufficient, no manual SUMO install or PATH configuration is needed.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


class SumoNotFoundError(RuntimeError):
    """Raised when the bundled SUMO distribution cannot be located."""


@lru_cache(maxsize=1)
def get_sumo_home() -> Path:
    """Return the root directory of the bundled SUMO distribution (contains bin/, tools/).

    Also sets the `SUMO_HOME` environment variable if it is not already set, because
    `sumolib`/`traci` and the SUMO command-line tools (netconvert, netgenerate,
    randomTrips.py) consult it internally.
    """
    try:
        import sumo as sumo_package
    except ImportError as exc:  # pragma: no cover - environment misconfiguration
        raise SumoNotFoundError(
            "The 'eclipse-sumo' package is not installed. Install project dependencies "
            "with `uv sync` (see pyproject.toml)."
        ) from exc

    home = Path(sumo_package.__file__).resolve().parent
    if not (home / "bin").is_dir() or not (home / "tools").is_dir():
        raise SumoNotFoundError(
            f"'sumo' package found at {home} but it does not look like a SUMO "
            "distribution (missing bin/ or tools/)."
        )

    os.environ.setdefault("SUMO_HOME", str(home))
    return home


def get_binary_path(name: str) -> Path:
    """Return the path to a SUMO binary (e.g. "sumo", "netgenerate", "netconvert",
    "duarouter"). Appends the platform executable suffix automatically.
    """
    home = get_sumo_home()
    suffix = ".exe" if os.name == "nt" else ""
    binary = home / "bin" / f"{name}{suffix}"
    if not binary.is_file():
        raise SumoNotFoundError(f"SUMO binary '{name}' not found at {binary}")
    return binary


def get_tool_path(relative_path: str) -> Path:
    """Return the path to a SUMO Python tool script under tools/, e.g. "randomTrips.py"."""
    home = get_sumo_home()
    tool = home / "tools" / relative_path
    if not tool.is_file():
        raise SumoNotFoundError(f"SUMO tool '{relative_path}' not found at {tool}")
    return tool
