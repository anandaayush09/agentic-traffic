from __future__ import annotations

from atlas.common.sumo_locate import get_binary_path, get_sumo_home, get_tool_path


def test_get_sumo_home_returns_dir_with_bin_and_tools() -> None:
    home = get_sumo_home()

    assert home.is_dir()
    assert (home / "bin").is_dir()
    assert (home / "tools").is_dir()


def test_get_sumo_home_sets_environment_variable(monkeypatch) -> None:
    monkeypatch.delenv("SUMO_HOME", raising=False)
    get_sumo_home.cache_clear()

    home = get_sumo_home()

    import os

    assert os.environ["SUMO_HOME"] == str(home)
    get_sumo_home.cache_clear()


def test_get_binary_path_finds_netgenerate() -> None:
    path = get_binary_path("netgenerate")

    assert path.is_file()


def test_get_tool_path_finds_random_trips() -> None:
    path = get_tool_path("randomTrips.py")

    assert path.is_file()
