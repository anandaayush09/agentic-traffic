"""Validates the quota/health-aware router: which provider it picks and why, how it reacts to
429s, bad keys and failures, that it always ends at the deterministic mapper instead of
raising, and that two chains sharing one Redis share budgets and cooldowns. Uses fake
providers and a fake clock -- the real HTTP providers are covered in test_llm_providers.py.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import fakeredis
import pytest
from langgraph.checkpoint.memory import InMemorySaver

from atlas.cognitive.llm.limits import (
    InMemoryRateLimitStore,
    ProviderLimits,
    RateLimitStore,
    RedisRateLimitStore,
)
from atlas.cognitive.llm.router import AdaptiveProviderChain, RoutedProvider
from atlas.cognitive.playbooks import PlaybookProposal, PlaybookRegistry
from atlas.cognitive.provider_chain import (
    ProviderAuthError,
    ProviderRateLimitedError,
    ProviderUnavailableError,
)
from atlas.cognitive.supervisor import build_supervisor_graph, initial_state
from atlas.cognitive.what_if import WhatIfResult

INCIDENT: dict[str, object] = {
    "incident_id": "inc-1",
    "class": "accident",
    "location": {"link_ids": ["42430342"]},
}
PLAYBOOKS = PlaybookRegistry().names()
GROQ_LIMITS = ProviderLimits(30, 14400)
GEMINI_LIMITS = ProviderLimits(10, 250)


def _proposal(tag: str) -> PlaybookProposal:
    return PlaybookProposal(
        playbook_name="hold_current_plan", params={"reason": tag}, rationale=f"from {tag}"
    )


class FakeProvider:
    def __init__(self, tag: str, error: Exception | None = None) -> None:
        self.tag = tag
        self.error = error
        self.calls = 0

    def propose(
        self, incident: dict[str, object], available_playbooks: list[str]
    ) -> PlaybookProposal:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return _proposal(self.tag)


class Clock:
    def __init__(self) -> None:
        self.now = 1_700_000_000.0

    def __call__(self) -> float:
        return self.now


def _chain(
    groq: FakeProvider,
    gemini: FakeProvider,
    *,
    groq_limits: ProviderLimits = GROQ_LIMITS,
    gemini_limits: ProviderLimits = GEMINI_LIMITS,
    store: RateLimitStore | None = None,
    clock: Clock | None = None,
) -> tuple[AdaptiveProviderChain, Clock]:
    clock = clock or Clock()
    chain = AdaptiveProviderChain(
        [
            RoutedProvider("groq", groq, groq_limits, prior_latency_ms=400.0),
            RoutedProvider("gemini", gemini, gemini_limits, prior_latency_ms=1500.0),
        ],
        store or InMemoryRateLimitStore(now_fn=clock),
        now_fn=clock,
    )
    return chain, clock


def _ask(chain: AdaptiveProviderChain) -> str:
    return chain.propose(INCIDENT, PLAYBOOKS)[1]


class TestChoosingTheProvider:
    def test_with_both_healthy_the_faster_one_answers(self) -> None:
        groq, gemini = FakeProvider("groq"), FakeProvider("gemini")
        chain, _ = _chain(groq, gemini)

        proposal, name = chain.propose(INCIDENT, PLAYBOOKS)

        assert name == "groq"
        assert proposal.rationale == "from groq"
        assert gemini.calls == 0

    def test_traffic_shifts_to_the_provider_with_more_headroom_as_budget_drains(self) -> None:
        groq, gemini = FakeProvider("groq"), FakeProvider("gemini")
        chain, _ = _chain(groq, gemini, groq_limits=ProviderLimits(4, 14400))

        answered_by = [_ask(chain) for _ in range(4)]

        # groq answers while it has most of its minute budget, then gemini takes over.
        assert answered_by[0] == "groq"
        assert answered_by[-1] == "gemini"
        assert groq.calls + gemini.calls == 4

    def test_a_provider_that_keeps_failing_sinks_in_the_ranking(self) -> None:
        groq = FakeProvider("groq", error=ProviderUnavailableError("junk output"))
        gemini = FakeProvider("gemini")
        chain, clock = _chain(groq, gemini)

        _ask(chain)  # groq fails, gemini answers
        clock.now += 11  # past groq's short error cooldown
        status = {s.name: s for s in chain.status()}

        assert status["groq"].reliability < 1.0
        assert status["groq"].cooldown_s == 0.0
        assert chain.status()[0].name == "gemini"  # lower reliability ranks it behind

    def test_status_lists_every_provider_best_first(self) -> None:
        chain, _ = _chain(FakeProvider("groq"), FakeProvider("gemini"))

        status = chain.status()

        assert [s.name for s in status] == ["groq", "gemini"]
        assert all(s.available and s.headroom == 1.0 for s in status)


class TestFailureHandling:
    def test_rate_limited_falls_through_in_the_same_call_then_cools_down(self) -> None:
        groq = FakeProvider("groq", error=ProviderRateLimitedError("429", retry_after_s=30.0))
        gemini = FakeProvider("gemini")
        chain, clock = _chain(groq, gemini)

        assert _ask(chain) == "gemini"  # groq said 429, gemini answered this same request
        assert _ask(chain) == "gemini"
        assert groq.calls == 1  # not hammered again while cooling down

        clock.now += 31
        status = {s.name: s for s in chain.status()}
        # Eligible again once the cooldown passes, though still ranked behind gemini
        # until it earns back its reliability.
        assert status["groq"].available
        assert chain.status()[0].name == "gemini"

    def test_a_provider_that_failed_is_forgiven_after_a_quiet_spell(self) -> None:
        groq = FakeProvider("groq", error=ProviderUnavailableError("blip"))
        chain, clock = _chain(groq, FakeProvider("gemini"))
        _ask(chain)
        assert {s.name: s for s in chain.status()}["groq"].reliability < 1.0

        clock.now += 601

        assert {s.name: s for s in chain.status()}["groq"].reliability == 1.0

    def test_a_rate_limit_without_a_hint_uses_a_default_cooldown(self) -> None:
        groq = FakeProvider("groq", error=ProviderRateLimitedError("429"))
        chain, _ = _chain(groq, FakeProvider("gemini"))

        _ask(chain)

        cooldown = {s.name: s.cooldown_s for s in chain.status()}["groq"]
        assert 0 < cooldown <= 60

    def test_a_rejected_key_is_benched_for_a_long_time(self) -> None:
        groq = FakeProvider("groq", error=ProviderAuthError("bad key"))
        chain, clock = _chain(groq, FakeProvider("gemini"))

        _ask(chain)
        clock.now += 600

        status = {s.name: s for s in chain.status()}
        assert status["groq"].cooldown_s > 0  # still benched ten minutes later
        assert not status["groq"].available

    def test_a_provider_out_of_budget_is_skipped(self) -> None:
        groq, gemini = FakeProvider("groq"), FakeProvider("gemini")
        chain, _ = _chain(
            groq, gemini, groq_limits=ProviderLimits(1, 100), gemini_limits=ProviderLimits(1, 100)
        )

        names = [_ask(chain) for _ in range(3)]

        assert names == ["groq", "gemini", "deterministic"]

    def test_when_every_provider_fails_the_deterministic_mapper_answers(self) -> None:
        groq = FakeProvider("groq", error=ProviderUnavailableError("down"))
        gemini = FakeProvider("gemini", error=ProviderRateLimitedError("429"))
        chain, _ = _chain(groq, gemini)

        proposal, name = chain.propose(INCIDENT, PLAYBOOKS)

        assert name == "deterministic"
        assert proposal.playbook_name == "divert_arterial"  # the mapper's accident default

    def test_no_providers_at_all_is_just_the_deterministic_mapper(self) -> None:
        chain = AdaptiveProviderChain([], InMemoryRateLimitStore())

        assert chain.propose(INCIDENT, PLAYBOOKS)[1] == "deterministic"

    def test_provider_names_must_be_unique(self) -> None:
        provider = RoutedProvider("x", FakeProvider("x"), ProviderLimits(1, 1))

        with pytest.raises(ValueError, match="unique"):
            AdaptiveProviderChain([provider, provider], InMemoryRateLimitStore())


class TestSharedRedisState:
    def test_two_chains_on_one_redis_share_budgets_and_cooldowns(self) -> None:
        """Two processes (the API and a script) must see one set of counters."""
        client = fakeredis.FakeRedis(decode_responses=True)
        clock = Clock()
        one, _ = _chain(
            FakeProvider("groq"),
            FakeProvider("gemini"),
            groq_limits=ProviderLimits(1, 100),
            store=RedisRateLimitStore(client),
            clock=clock,
        )
        two, _ = _chain(
            FakeProvider("groq"),
            FakeProvider("gemini"),
            groq_limits=ProviderLimits(1, 100),
            store=RedisRateLimitStore(client),
            clock=clock,
        )

        first = _ask(one)  # uses groq's only request this minute

        # The second process sees that budget already spent, so it goes to gemini.
        assert first == "groq"
        assert {s.name: s for s in two.status()}["groq"].headroom == 0.0
        assert _ask(two) == "gemini"


class TestSupervisorIntegration:
    def test_the_supervisor_uses_the_adaptive_chain_and_records_who_answered(self) -> None:
        chain, _ = _chain(FakeProvider("groq"), FakeProvider("gemini"))
        graph = build_supervisor_graph(
            chain,
            PlaybookRegistry(),
            Path("unused.net.xml"),
            Path("unused.rou.xml"),
            InMemorySaver(),
        )
        unchecked = WhatIfResult(
            checked=False, improved=None, baseline_mean_pressure=None, proposed_mean_pressure=None
        )

        with patch("atlas.cognitive.supervisor.run_what_if", return_value=unchecked):
            state = graph.invoke(
                initial_state(dict(INCIDENT)), config={"configurable": {"thread_id": "llm-1"}}
            )

        assert state["provider_used"] == "groq"
        assert state["proposal"]["rationale"] == "from groq"
