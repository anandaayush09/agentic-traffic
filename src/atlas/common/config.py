"""Configuration loading for ATLAS.

Two kinds of configuration exist in this codebase, deliberately kept separate:

- **Declarative YAML configs** under `configs/` (network definitions, training
  hyperparameters, scenario descriptions) -- versioned in git/DVC, the same for every
  environment, loaded via `load_yaml_model`.
- **Process/environment settings** (log level, secrets, connection strings) -- vary per
  deployment, read from environment variables via `pydantic-settings`, never committed.

This split is what Part 7.7 of the design docs means by "Pydantic v2 ... schema
validation shared between API models and bus schemas": every config surface in ATLAS is a
validated Pydantic model, not a loose dict.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigError(RuntimeError):
    """Raised when a YAML config file is missing or fails schema validation."""


def load_yaml_model[T: BaseModel](path: Path, model: type[T]) -> T:
    """Load a YAML file and validate it against a Pydantic model.

    Raises `ConfigError` (not the raw `pydantic.ValidationError`/`OSError`) so callers
    and tests can catch one exception type regardless of what went wrong -- a missing
    file and a malformed field are both "this config is unusable", and the message tells
    the operator which.
    """
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    try:
        return model.model_validate(raw)
    except Exception as exc:  # pydantic.ValidationError, re-raised with file context
        raise ConfigError(f"Invalid config in {path}: {exc}") from exc


class ProjectSettings(BaseSettings):
    """Process-level settings, overridable via `ATLAS_*` environment variables (see
    `.env.example`). This is where deployment-specific values live -- nothing here is
    committed with real values.
    """

    model_config = SettingsConfigDict(env_prefix="ATLAS_", env_file=".env", extra="ignore")

    log_level: str = "INFO"
    log_format: Literal["console", "json"] = "console"


def project_root() -> Path:
    """Return the repository root, resolved from this file's location so it works
    regardless of the caller's current working directory.
    """
    return Path(__file__).resolve().parents[3]
