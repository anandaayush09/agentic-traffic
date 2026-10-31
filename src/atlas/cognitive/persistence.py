"""Durable state for the cognitive plane: incident decisions survive a restart.

`InMemorySaver` (the default) loses every incident -- including ones paused waiting for an
operator's approval -- when the API process restarts. LangGraph's `SqliteSaver` checkpoints
each graph step to a SQLite file, so a paused incident resumes with `Command(resume=...)`
after a restart exactly as if nothing happened (docs/design/04_agent_architecture.md section
4.4: "LangGraph durable checkpoints -- survives restart mid-incident"). The checkpointer
stores *graph state per thread*, not a list of threads, so the API also needs a durable
index of which incident ids exist -- `SqliteIncidentIndex`, in the same database file.

SQLite, not Postgres (ADR-0003's durable tier): a single-node demo/control room needs no
server, and the swap point is the same -- `IncidentIndex` is a protocol and the checkpointer
is whatever `build_supervisor_graph` is handed.

Scope, said plainly: this persists the *decision record* (incident, proposal, what-if
evidence, approval). It does NOT persist the running traffic simulation -- after a restart
the live twin starts from t=0 with no disruptions, even though an old incident may read
"cleared" or "pending". The incident log is history, not a snapshot of the road.
"""

from __future__ import annotations

# Windows native-DLL collision, verified: SUMO ships its own sqlite3.dll, and if Python's
# `sqlite3` module loads first, libsumo can no longer start (0xc0000139, entry point not
# found) -- while "libsumo first, then sqlite3" works. This module is imported by the API,
# which later starts libsumo (the what-if tool), so load libsumo before anything sqlite.
# The same fix as "import libsumo before sumolib/pyproj" elsewhere in this project.
import contextlib

# Only a *missing* SUMO is tolerated; a DLL load failure (a plain ImportError) must stay loud.
with contextlib.suppress(ModuleNotFoundError):
    import libsumo  # noqa: F401

import sqlite3  # noqa: E402
import threading  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Protocol  # noqa: E402

from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402


class IncidentIndex(Protocol):
    def add(self, incident_id: str) -> None: ...
    def contains(self, incident_id: str) -> bool: ...
    def all(self) -> list[str]: ...


class InMemoryIncidentIndex:
    def __init__(self) -> None:
        self._ids: set[str] = set()

    def add(self, incident_id: str) -> None:
        self._ids.add(incident_id)

    def contains(self, incident_id: str) -> bool:
        return incident_id in self._ids

    def all(self) -> list[str]:
        return sorted(self._ids)


class SqliteIncidentIndex:
    """Incident ids in a SQLite table. Safe to share across the API's worker threads (one
    connection, guarded by a lock -- `check_same_thread=False` alone is not enough)."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS incident_index ("
                "incident_id TEXT PRIMARY KEY, "
                "created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            )
            self._conn.commit()

    def add(self, incident_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO incident_index (incident_id) VALUES (?)", (incident_id,)
            )
            self._conn.commit()

    def contains(self, incident_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM incident_index WHERE incident_id = ?", (incident_id,)
            ).fetchone()
        return row is not None

    def all(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT incident_id FROM incident_index ORDER BY incident_id"
            ).fetchall()
        return [str(r[0]) for r in rows]


@dataclass
class DurableState:
    checkpointer: SqliteSaver
    index: SqliteIncidentIndex
    path: Path
    _conn: sqlite3.Connection

    def close(self) -> None:
        self._conn.close()


def open_durable_state(path: Path) -> DurableState:
    """Opens (creating if needed) the SQLite file backing graph checkpoints and the incident
    index. WAL mode so readers (listing incidents) don't block the writer (a graph step)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    return DurableState(
        checkpointer=SqliteSaver(conn), index=SqliteIncidentIndex(conn), path=path, _conn=conn
    )
