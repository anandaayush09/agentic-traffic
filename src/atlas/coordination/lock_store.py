"""Distributed lock protocol for the Preemption Overlay (docs/design/
04_agent_architecture.md section 4.3: "Locks via Redis (atomic SETNX with TTL -- a lock
leak self-heals)"). `LockStore` is the swap point -- `InMemoryLockStore` is the interim,
correct-for-a-single-process implementation used until Redis actually lands, mirroring
the exact pattern already established for `atlas.bus.dedup.MessageDeduplicator`
(Redis isn't wired up yet; see the ATLAS dev-environment memory). A Redis-backed
implementation (`SET key holder NX EX ttl_s`) is a drop-in replacement behind the same
three-method interface, not a redesign.

TTL correctness is the entire point of this abstraction: a lock that isn't explicitly
released still expires on its own after `ttl_s`, so a Preemption Overlay process that
dies mid-run can never strand a junction under permanent lock (docs/design/
04_agent_architecture.md section 4.3's "Failure recovery: TTL'd locks guarantee
junctions return to normal control even if the overlay dies mid-run"). `now_fn` is
injectable specifically so this self-healing property is testable with a fake clock
(the Week 7 "lock-TTL recovery chaos test"), not real sleeps.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Protocol


class LockStore(Protocol):
    def acquire(self, key: str, holder: str, ttl_s: float) -> bool: ...
    def release(self, key: str, holder: str) -> None: ...
    def holder(self, key: str) -> str | None: ...


class InMemoryLockStore:
    """Process-local, in-memory SETNX-with-TTL. Correct within one process's lifetime
    (the same scope limitation `InMemoryDeduplicator` documents) -- a Redis-backed
    version is needed once the Preemption Overlay and the junctions it locks run as
    separate processes sharing state.
    """

    def __init__(self, now_fn: Callable[[], datetime] | None = None) -> None:
        self._now_fn = now_fn or (lambda: datetime.now(UTC))
        self._locks: dict[str, tuple[str, datetime]] = {}

    def acquire(self, key: str, holder: str, ttl_s: float) -> bool:
        """Atomic (single dict write, no intervening I/O) SETNX: acquires if the key is
        unheld or its lock has expired, or if `holder` already holds it (idempotent
        re-acquire/renew, refreshing the TTL). Refuses if a DIFFERENT holder's lock is
        still live -- exactly the first-lock-wins semantics the design's conflicting-
        emergency-route resolution depends on.
        """
        current_holder = self.holder(key)
        if current_holder is not None and current_holder != holder:
            return False
        self._locks[key] = (holder, self._now_fn() + timedelta(seconds=ttl_s))
        return True

    def release(self, key: str, holder: str) -> None:
        """Only releases if `holder` is the actual current holder -- a stale/late
        release from a run that already lost the lock (e.g. to expiry, then someone
        else acquired it) must never release a DIFFERENT run's live lock.
        """
        entry = self._locks.get(key)
        if entry is not None and entry[0] == holder:
            del self._locks[key]

    def holder(self, key: str) -> str | None:
        entry = self._locks.get(key)
        if entry is None:
            return None
        holder, expires_at = entry
        if self._now_fn() >= expires_at:
            del self._locks[key]  # expired -- self-heals, matching a real Redis TTL
            return None
        return holder
