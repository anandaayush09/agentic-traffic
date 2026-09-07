"""Validates TokenBucket's refill arithmetic (hand-verified with a fake clock) and
ProviderChain's fallback ordering -- the real behavior the Week 9 "provider-fallback
test (simulate Groq exhaustion -> falls through chain correctly)" asks for, using fake
providers rather than real Groq/Gemini/Ollama calls (no API keys available here; see
provider_chain.py's own module docstring for why real provider classes aren't built).
"""

from __future__ import annotations

import pytest

from atlas.cognitive.playbooks import PlaybookProposal
from atlas.cognitive.provider_chain import (
    ProviderChain,
    ProviderError,
    ProviderRateLimitedError,
    ProviderUnavailableError,
    TokenBucket,
)

INCIDENT = {"incident_id": "inc-1", "class": "accident", "status": "detected", "confidence": 0.9}


class _FakeProvider:
    """Always returns a fixed proposal naming itself, or always raises the given
    error -- deterministic, controllable stand-ins for a real LLM provider."""

    def __init__(self, name: str, error: ProviderError | None = None) -> None:
        self.name = name
        self.error = error
        self.call_count = 0

    def propose(
        self, incident: dict[str, object], available_playbooks: list[str]
    ) -> PlaybookProposal:
        self.call_count += 1
        if self.error is not None:
            raise self.error
        return PlaybookProposal(
            playbook_name="hold_current_plan",
            params={"reason": "test"},
            rationale=f"proposed by {self.name}",
        )


def _bucket(capacity: float = 10.0) -> TokenBucket:
    return TokenBucket(capacity=capacity, refill_rate_per_s=1.0)


class TestTokenBucket:
    def test_starts_full_and_can_consume_up_to_capacity(self) -> None:
        bucket = TokenBucket(capacity=3.0, refill_rate_per_s=1.0)
        assert bucket.try_consume() is True
        assert bucket.try_consume() is True
        assert bucket.try_consume() is True
        assert bucket.try_consume() is False  # exhausted

    def test_refills_over_time_hand_computed(self) -> None:
        """capacity=5, refill_rate=2/s, starting empty (consume all 5 immediately),
        then advance the fake clock by 1.5s -> 3.0 tokens refilled (capped at capacity).
        """
        now = {"t": 0.0}
        bucket = TokenBucket(capacity=5.0, refill_rate_per_s=2.0, now_fn=lambda: now["t"])
        for _ in range(5):
            assert bucket.try_consume() is True
        assert bucket.try_consume() is False

        now["t"] = 1.5
        assert bucket.try_consume() is True  # 3.0 tokens available now
        assert bucket.try_consume() is True
        assert bucket.try_consume() is True
        assert bucket.try_consume() is False  # only 3 were available

    def test_refill_never_exceeds_capacity(self) -> None:
        now = {"t": 0.0}
        bucket = TokenBucket(capacity=2.0, refill_rate_per_s=100.0, now_fn=lambda: now["t"])
        now["t"] = 1000.0  # would refill far past capacity without the cap

        assert bucket.try_consume() is True
        assert bucket.try_consume() is True
        assert bucket.try_consume() is False

    def test_rejects_non_positive_capacity_or_rate(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            TokenBucket(capacity=0, refill_rate_per_s=1.0)
        with pytest.raises(ValueError, match="positive"):
            TokenBucket(capacity=1.0, refill_rate_per_s=0)


class TestProviderChainFallback:
    def test_first_provider_succeeds_and_is_the_only_one_called(self) -> None:
        first = _FakeProvider("groq")
        second = _FakeProvider("gemini")
        chain = ProviderChain([("groq", first, _bucket()), ("gemini", second, _bucket())])

        proposal, provider_name = chain.propose(INCIDENT, ["hold_current_plan"])

        assert provider_name == "groq"
        assert "groq" in proposal.rationale
        assert first.call_count == 1
        assert second.call_count == 0

    def test_falls_through_on_provider_error_to_the_next_provider(self) -> None:
        """The literal Week 9 ask: simulate a provider being unavailable/exhausted and
        confirm the chain falls through correctly."""
        first = _FakeProvider("groq", error=ProviderUnavailableError("down"))
        second = _FakeProvider("gemini")
        chain = ProviderChain([("groq", first, _bucket()), ("gemini", second, _bucket())])

        proposal, provider_name = chain.propose(INCIDENT, ["hold_current_plan"])

        assert provider_name == "gemini"
        assert "gemini" in proposal.rationale

    def test_falls_through_on_exhausted_rate_limit_without_calling_the_provider(self) -> None:
        """A rate-limited provider must not even be CALLED -- the whole point of the
        token bucket is failing fast client-side instead of hitting a real 429."""
        exhausted_bucket = TokenBucket(capacity=1.0, refill_rate_per_s=0.001)
        exhausted_bucket.try_consume()  # drain it before the chain ever runs
        first = _FakeProvider("groq")
        second = _FakeProvider("gemini")
        chain = ProviderChain([("groq", first, exhausted_bucket), ("gemini", second, _bucket())])

        proposal, provider_name = chain.propose(INCIDENT, ["hold_current_plan"])

        assert provider_name == "gemini"
        assert first.call_count == 0  # never called -- rejected by the bucket alone

    def test_every_provider_failing_falls_through_to_the_deterministic_mapper(self) -> None:
        """The full chain: every real provider exhausted/down -- the deterministic
        mapper must still produce a valid answer, never raise."""
        first = _FakeProvider("groq", error=ProviderRateLimitedError("429"))
        second = _FakeProvider("gemini", error=ProviderUnavailableError("timeout"))
        third = _FakeProvider("ollama", error=ProviderUnavailableError("not running"))
        chain = ProviderChain(
            [
                ("groq", first, _bucket()),
                ("gemini", second, _bucket()),
                ("ollama", third, _bucket()),
            ]
        )

        proposal, provider_name = chain.propose(INCIDENT, ["hold_current_plan"])

        assert provider_name == "deterministic"
        assert (
            proposal.playbook_name == "divert_arterial"
        )  # the real mapper's answer for "accident"

    def test_empty_provider_list_still_produces_a_deterministic_answer(self) -> None:
        chain = ProviderChain([])

        proposal, provider_name = chain.propose(INCIDENT, ["hold_current_plan"])

        assert provider_name == "deterministic"
