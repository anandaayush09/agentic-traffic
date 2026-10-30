"""AdaptiveProviderChain: picks the most suitable LLM provider for each incident.

`ProviderChain` tries providers in a fixed order. With two free-tier providers whose quotas
are shared across processes and whose health changes, the right question per incident is
"which one should answer *now*?" -- so this chain ranks them on live state from the
`RateLimitStore` (Redis):

    score = headroom x reliability

- **headroom** (0..1): the unused fraction of the tighter of the provider's per-minute and
  per-day request budgets. As one provider's budget drains, traffic shifts to the other.
- **reliability** (0..1): moving-average success rate, so a provider that keeps failing or
  returning unusable replies sinks in the ranking.

Providers whose score is within 0.05 of each other count as tied and the lower observed
latency wins (Groq is typically much faster than Gemini, so with both healthy it answers
first). Providers on cooldown (after a 429 -- honouring Retry-After -- or an auth failure
or another error) or out of budget are skipped outright.

For the chosen provider a request is *reserved* against the shared counters before the
call (so concurrent processes can't overshoot), then the outcome updates health stats and
any cooldown. If every LLM provider is unavailable the deterministic mapper answers --
`propose()` never raises, exactly like `ProviderChain`.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from atlas.cognitive.deterministic_mapper import map_incident_to_playbook
from atlas.cognitive.llm.limits import ProviderLimits, QuotaTracker, RateLimitStore
from atlas.cognitive.playbooks import PlaybookProposal
from atlas.cognitive.provider_chain import (
    PlaybookProvider,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitedError,
)
from atlas.common.logging import get_logger

logger = get_logger(__name__)

_TIE_BAND = 0.05
_DEFAULT_RATE_LIMIT_COOLDOWN_S = 60.0
_AUTH_COOLDOWN_S = 3600.0
_ERROR_COOLDOWN_S = 10.0


@dataclass(frozen=True, slots=True)
class RoutedProvider:
    name: str
    provider: PlaybookProvider
    limits: ProviderLimits
    prior_latency_ms: float = 1000.0
    """Assumed latency until a real one has been measured -- only used to break ties."""


@dataclass(frozen=True, slots=True)
class ProviderStatus:
    name: str
    headroom: float
    reliability: float
    latency_ms: float
    cooldown_s: float
    score: float
    available: bool


class AdaptiveProviderChain:
    def __init__(
        self,
        providers: Sequence[RoutedProvider],
        store: RateLimitStore,
        now_fn: Callable[[], float] | None = None,
        prefix: str = "atlas:llm",
    ) -> None:
        names = [p.name for p in providers]
        if len(set(names)) != len(names):
            raise ValueError(f"provider names must be unique, got {names}")
        self._providers = list(providers)
        self._now_fn = now_fn or time.time
        self._quota = QuotaTracker(store, now_fn=self._now_fn, prefix=prefix)

    @property
    def provider_names(self) -> list[str]:
        return [p.name for p in self._providers]

    @property
    def providers(self) -> list[RoutedProvider]:
        return list(self._providers)

    def status(self) -> list[ProviderStatus]:
        """Every provider's live routing inputs, best candidate first."""
        entries = [self._status_of(p) for p in self._providers]
        return sorted(entries, key=_ranking_key)

    def propose(
        self, incident: dict[str, object], available_playbooks: list[str]
    ) -> tuple[PlaybookProposal, str]:
        by_name = {p.name: p for p in self._providers}
        for status in self.status():
            if not status.available:
                continue
            routed = by_name[status.name]
            if not self._quota.try_reserve(routed.name, routed.limits):
                continue  # lost the race for the last slot to another process
            started = time.monotonic()
            try:
                proposal = routed.provider.propose(incident, available_playbooks)
            except ProviderRateLimitedError as exc:
                self._record_failure(routed.name, started)
                cooldown = exc.retry_after_s or _DEFAULT_RATE_LIMIT_COOLDOWN_S
                self._quota.start_cooldown(routed.name, cooldown)
                logger.warning("llm.rate_limited", provider=routed.name, cooldown_s=cooldown)
                continue
            except ProviderAuthError as exc:
                self._record_failure(routed.name, started)
                self._quota.start_cooldown(routed.name, _AUTH_COOLDOWN_S)
                logger.error("llm.auth_failed", provider=routed.name, error=str(exc))
                continue
            except ProviderError as exc:
                self._record_failure(routed.name, started)
                self._quota.start_cooldown(routed.name, _ERROR_COOLDOWN_S)
                logger.warning("llm.provider_failed", provider=routed.name, error=str(exc))
                continue
            self._quota.record_outcome(routed.name, True, (time.monotonic() - started) * 1000)
            return proposal, routed.name

        if self._providers:  # with none configured, the deterministic mapper is the plan
            logger.warning("llm.chain_exhausted", incident_class=incident.get("class"))
        return map_incident_to_playbook(incident), "deterministic"

    def _record_failure(self, name: str, started: float) -> None:
        self._quota.record_outcome(name, False, (time.monotonic() - started) * 1000)

    def _status_of(self, routed: RoutedProvider) -> ProviderStatus:
        headroom = self._quota.headroom(routed.name, routed.limits)
        stats = self._quota.stats(routed.name)
        cooldown = self._quota.cooldown_remaining(routed.name)
        latency = (
            stats.latency_ms_ema if stats.latency_ms_ema is not None else routed.prior_latency_ms
        )
        return ProviderStatus(
            name=routed.name,
            headroom=headroom,
            reliability=stats.ok_ema,
            latency_ms=latency,
            cooldown_s=cooldown,
            score=headroom * stats.ok_ema,
            available=cooldown <= 0.0 and headroom > 0.0,
        )


def _ranking_key(status: ProviderStatus) -> tuple[float, float]:
    # Bucket the score so near-equal providers tie, then prefer the faster one. Sort
    # ascending, so the bucket is negated (higher score first).
    return (-round(status.score / _TIE_BAND), status.latency_ms)
