"""The LLM provider chain: Groq -> Gemini -> Ollama -> deterministic mapper (docs/design/
04_agent_architecture.md section 4.4; docs/design/06_ai_pipeline.md section 6.6's
"Groq llama-3.1-8b-instant... primary; Gemini 3 Flash... secondary; Ollama... offline;
deterministic mapper terminal"). "Failure recovery: durable graph checkpoints; if the
LLM chain is exhausted/down, the deterministic mapper... keeps responses flowing".

Scope, deliberately: this module is the FALLBACK LOGIC itself -- token-bucket rate
limiting, ordered provider trial, and the always-succeeding deterministic terminal --
built and thoroughly tested against fake providers (`PlaybookProvider` is a Protocol,
same swap-point pattern already used for `LockStore`/`MessageDeduplicator`). Real Groq/
Gemini/Ollama client classes are NOT implemented here: unlike the YOLO/ONNX pipeline
(Week 8), where a real local smoke test could verify the exact Ultralytics API calls
before writing against them, there is no way to verify a real LLM provider's API
surface in this environment without an actual API key and a live network call -- and
guessing at a specific SDK's exact call shape without any way to check it would be
exactly the kind of unverified code this project has consistently avoided shipping.
Wiring in a real provider is implementing `PlaybookProvider` (one `propose` method) for
that provider's actual client library -- a small, isolated task deferred until it can
be verified against a real key, not a redesign of anything here.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Protocol

from atlas.cognitive.deterministic_mapper import map_incident_to_playbook
from atlas.cognitive.playbooks import PlaybookProposal
from atlas.common.logging import get_logger

logger = get_logger(__name__)


class ProviderError(Exception):
    """Base class for a provider being unable to produce a proposal right now --
    caught by ProviderChain to fall through to the next provider, never propagated."""


class ProviderRateLimitedError(ProviderError):
    """The provider's own rate limit was hit -- distinct from a hard failure so
    callers/logs can tell "will work again later" apart from "broken right now"."""


class ProviderUnavailableError(ProviderError):
    """The provider is down, timed out, or returned something not usable (e.g. failed
    structured-JSON-output parsing)."""


class PlaybookProvider(Protocol):
    def propose(
        self, incident: dict[str, object], available_playbooks: list[str]
    ) -> PlaybookProposal: ...


class TokenBucket:
    """A standard token bucket: `capacity` tokens, refilling at `refill_rate_per_s`,
    each `try_consume()` call costing one token. Used to enforce a provider's own free-
    tier rate limit client-side (docs/design/06_ai_pipeline.md section 6.6's "token
    buckets") -- failing fast and falling through to the next provider is much cheaper
    than waiting on a 429 from the real API. `now_fn` is injectable so refill timing is
    testable with a fake clock, the same pattern as `InMemoryLockStore`.
    """

    def __init__(
        self, capacity: float, refill_rate_per_s: float, now_fn: Callable[[], float] | None = None
    ) -> None:
        if capacity <= 0 or refill_rate_per_s <= 0:
            raise ValueError("capacity and refill_rate_per_s must both be positive")
        self._capacity = capacity
        self._refill_rate_per_s = refill_rate_per_s
        self._now_fn = now_fn or time.monotonic
        self._tokens = capacity
        self._last_refill = self._now_fn()

    def _refill(self) -> None:
        now = self._now_fn()
        elapsed_s = now - self._last_refill
        self._tokens = min(self._capacity, self._tokens + elapsed_s * self._refill_rate_per_s)
        self._last_refill = now

    def try_consume(self, cost: float = 1.0) -> bool:
        self._refill()
        if self._tokens < cost:
            return False
        self._tokens -= cost
        return True


class ProviderChain:
    """Tries each `(name, provider, token_bucket)` in order; falls through on
    `ProviderError` or an exhausted token bucket; the deterministic mapper is always
    tried last and never raises, so `propose()` itself never raises either --
    "the deterministic mapper keeps responses flowing" made literal.
    """

    def __init__(self, providers: list[tuple[str, PlaybookProvider, TokenBucket]]) -> None:
        self._providers = providers

    def propose(
        self, incident: dict[str, object], available_playbooks: list[str]
    ) -> tuple[PlaybookProposal, str]:
        """Returns (proposal, provider_name_that_succeeded) -- the name matters for
        audit/observability (which tier actually answered), not just the proposal.
        """
        for name, provider, bucket in self._providers:
            if not bucket.try_consume():
                logger.warning("provider_chain.rate_limited", provider=name)
                continue
            try:
                return provider.propose(incident, available_playbooks), name
            except ProviderError as exc:
                logger.warning("provider_chain.provider_failed", provider=name, error=str(exc))
                continue

        logger.warning("provider_chain.exhausted", incident_class=incident.get("class"))
        return map_incident_to_playbook(incident), "deterministic"
