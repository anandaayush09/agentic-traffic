from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from atlas.common.config import ConfigError, load_yaml_model, project_root


class _DummyModel(BaseModel):
    name: str
    count: int = 0


def test_load_yaml_model_success(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("name: hello\ncount: 3\n", encoding="utf-8")

    result = load_yaml_model(config_file, _DummyModel)

    assert result == _DummyModel(name="hello", count=3)


def test_load_yaml_model_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.yaml"

    with pytest.raises(ConfigError, match="not found"):
        load_yaml_model(missing, _DummyModel)


def test_load_yaml_model_invalid_yaml(tmp_path: Path) -> None:
    bad_file = tmp_path / "bad.yaml"
    bad_file.write_text("name: [unclosed\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="Invalid YAML"):
        load_yaml_model(bad_file, _DummyModel)


def test_load_yaml_model_schema_violation(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("count: not_a_number\n", encoding="utf-8")  # missing 'name' too

    with pytest.raises(ConfigError, match="Invalid config"):
        load_yaml_model(config_file, _DummyModel)


def test_project_root_contains_pyproject() -> None:
    root = project_root()

    assert (root / "pyproject.toml").is_file()
