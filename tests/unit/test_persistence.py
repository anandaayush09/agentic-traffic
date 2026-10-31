"""Validates durable cognitive state: the incident index contract, and the headline property
-- an incident paused for operator approval survives the API process "restarting" (a brand
new connection, graph and service on the same SQLite file) and can still be approved.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from atlas.cognitive.persistence import (
    InMemoryIncidentIndex,
    SqliteIncidentIndex,
    open_durable_state,
)
from atlas.cognitive.playbooks import PlaybookRegistry
from atlas.cognitive.provider_chain import ProviderChain
from atlas.cognitive.supervisor import build_supervisor_graph
from atlas.cognitive.what_if import WhatIfResult
from atlas.services.cognitive_api import CognitiveApiService, IncidentNotFoundError

UNCHECKED = WhatIfResult(
    checked=False, improved=None, baseline_mean_pressure=None, proposed_mean_pressure=None
)
AUTO_APPLY = WhatIfResult(
    checked=True, improved=True, baseline_mean_pressure=2.0, proposed_mean_pressure=1.0
)


def _incident(incident_id: str, incident_class: str) -> dict[str, object]:
    return {
        "incident_id": incident_id,
        "class": incident_class,
        "status": "detected",
        "confidence": 0.9,
        "location": {"link_ids": ["42430342"]},
    }


def _service(db: Path) -> tuple[CognitiveApiService, object]:
    """A fresh 'process': new connection, new compiled graph, new service."""
    state = open_durable_state(db)
    graph = build_supervisor_graph(
        ProviderChain([]),
        PlaybookRegistry(),
        Path("unused.net.xml"),
        Path("unused.rou.xml"),
        state.checkpointer,
    )
    return CognitiveApiService(graph, index=state.index), state


class TestIncidentIndex:
    @pytest.fixture(params=["memory", "sqlite"])
    def index(self, request: pytest.FixtureRequest):  # type: ignore[no-untyped-def]
        if request.param == "memory":
            return InMemoryIncidentIndex()
        return SqliteIncidentIndex(sqlite3.connect(":memory:", check_same_thread=False))

    def test_add_contains_and_sorted_listing(self, index) -> None:  # type: ignore[no-untyped-def]
        index.add("b")
        index.add("a")

        assert index.contains("a")
        assert not index.contains("zzz")
        assert index.all() == ["a", "b"]

    def test_adding_twice_is_harmless(self, index) -> None:  # type: ignore[no-untyped-def]
        index.add("a")
        index.add("a")

        assert index.all() == ["a"]

    def test_sqlite_index_survives_reopening_the_file(self, tmp_path: Path) -> None:
        db = tmp_path / "idx.sqlite"
        first = SqliteIncidentIndex(sqlite3.connect(db))
        first.add("kept")

        second = SqliteIncidentIndex(sqlite3.connect(db))

        assert second.all() == ["kept"]


class TestRestartSurvival:
    def test_an_incident_paused_for_approval_survives_a_restart_and_can_be_approved(
        self, tmp_path: Path
    ) -> None:
        db = tmp_path / "state" / "atlas.sqlite"  # the parent directory is created for us
        before, state_before = _service(db)
        with patch("atlas.cognitive.supervisor.run_what_if", return_value=UNCHECKED):
            submitted = before.submit_incident(_incident("inc-pending", "accident"))
        assert submitted["pending_approval"] is True
        state_before.close()  # type: ignore[attr-defined]

        after, state_after = _service(db)  # "the API restarted"
        listed = after.list_incidents()

        assert [i["incident_id"] for i in listed] == ["inc-pending"]
        assert listed[0]["pending_approval"] is True
        assert listed[0]["interrupt"]["question"] == "Approve this playbook proposal?"
        approved = after.resolve_approval("inc-pending", approved=True)
        assert approved["state"]["status"] == "cleared"
        state_after.close()  # type: ignore[attr-defined]

        final, state_final = _service(db)  # and the decision itself is durable too
        assert final.get_incident("inc-pending")["state"]["status"] == "cleared"
        state_final.close()  # type: ignore[attr-defined]

    def test_resolved_and_auto_applied_incidents_keep_their_full_record(
        self, tmp_path: Path
    ) -> None:
        db = tmp_path / "atlas.sqlite"
        before, state_before = _service(db)
        with patch("atlas.cognitive.supervisor.run_what_if", return_value=AUTO_APPLY):
            before.submit_incident(_incident("inc-auto", "sensor_fault"))
        state_before.close()  # type: ignore[attr-defined]

        after, state_after = _service(db)
        record = after.get_incident("inc-auto")

        assert record["state"]["auto_applied"] is True
        assert record["state"]["proposal"]["playbook_name"] == "switch_to_max_pressure"
        assert record["pending_approval"] is False
        state_after.close()  # type: ignore[attr-defined]

    def test_an_unknown_incident_is_still_unknown_after_a_restart(self, tmp_path: Path) -> None:
        db = tmp_path / "atlas.sqlite"
        _, state = _service(db)
        state.close()  # type: ignore[attr-defined]

        again, state_again = _service(db)

        with pytest.raises(IncidentNotFoundError):
            again.get_incident("never-submitted")
        state_again.close()  # type: ignore[attr-defined]


class TestNativeLibraryOrder:
    def test_importing_persistence_does_not_break_libsumo_afterwards(self) -> None:
        """Regression: importing `sqlite3` before libsumo kills libsumo on Windows (SUMO
        ships its own sqlite3.dll). Run in a fresh interpreter -- this process may have
        loaded libsumo already, which would hide the bug."""
        src = Path(__file__).resolve().parents[2] / "src"
        code = "import atlas.cognitive.persistence, sqlite3; import libsumo; print('ok')"
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, "PYTHONPATH": str(src)},
        )

        assert result.returncode == 0, result.stderr[-500:]
        assert "ok" in result.stdout
