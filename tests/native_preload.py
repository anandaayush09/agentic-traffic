"""Loaded as an early pytest plugin (`-p tests.native_preload`, see pyproject addopts).

SUMO ships its own sqlite3.dll; if anything imports Python's `sqlite3` BEFORE libsumo, libsumo can no
longer start on Windows ("DLL load failed while importing _libsumo"). `coverage` stores its data in
SQLite and starts before conftest.py runs, so `pytest --cov` hit exactly that. Importing libsumo here,
before any other plugin gets a chance, makes the whole suite independent of plugin load order.
(tests/conftest.py keeps its own import for runs that bypass this plugin.)
"""

import contextlib

with contextlib.suppress(
    ModuleNotFoundError
):  # a missing SUMO is fine; a DLL failure must stay loud
    import libsumo  # noqa: F401
