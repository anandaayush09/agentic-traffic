"""At-least-once bus delivery means every consumer must be idempotent (docs/design/
05_communication.md section 5.6: "all consumers idempotent via msg_id dedup (Redis set
with TTL >= retention)"). `MessageDeduplicator` is the swap point: `InMemoryDeduplicator`
is the interim, correct-for-a-single-process implementation used until Redis actually
lands (Week 5-6's control plane) -- a Redis-backed implementation is a drop-in
replacement behind the same two-method interface, not a redesign.
"""

from __future__ import annotations

from typing import Protocol


class MessageDeduplicator(Protocol):
    def seen(self, msg_id: str) -> bool: ...
    def mark(self, msg_id: str) -> None: ...


class InMemoryDeduplicator:
    """Process-local, unbounded set -- correct within one consumer process's lifetime,
    not across restarts or multiple consumer instances. That gap is exactly what the
    Redis-backed version (TTL'd, shared across processes) closes later; not built now
    because nothing in this repo runs multiple consumer processes against the same topic
    yet.
    """

    def __init__(self) -> None:
        self._seen: set[str] = set()

    def seen(self, msg_id: str) -> bool:
        return msg_id in self._seen

    def mark(self, msg_id: str) -> None:
        self._seen.add(msg_id)
