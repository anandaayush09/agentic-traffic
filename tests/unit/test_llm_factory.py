"""Validates building the provider chain from settings/environment: which providers appear,
that keys never leak into notes or reprs, how an unreachable Redis is handled, and the
settings' env-var names and defaults.
"""

from __future__ import annotations

import json

import httpx
import pytest

from atlas.cognitive.llm.factory import LlmSettings, build_llm_chain
from atlas.cognitive.llm.limits import InMemoryRateLimitStore

GROQ_KEY = "gsk_test_SECRET_groq_key_123"
GEMINI_KEY = "AIza_test_SECRET_gemini_key_456"

INCIDENT: dict[str, object] = {
    "incident_id": "inc-1",
    "class": "accident",
    "status": "detected",
    "confidence": 0.9,
    "location": {"link_ids": ["42430342"]},
}


def _settings(**overrides: object) -> LlmSettings:
    # _env_file=None: never let a developer's real .env leak into a test.
    return LlmSettings(_env_file=None, **overrides)  # type: ignore[arg-type]


def _answering_client(seen: list[httpx.Request]) -> httpx.Client:
    reply = json.dumps(
        {
            "playbook_name": "divert_arterial",
            "params": {"from_edge": "42430342", "to_edge": "42430342", "duration_s": 600},
            "rationale": "test",
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if "groq" in request.url.host:
            return httpx.Response(200, json={"choices": [{"message": {"content": reply}}]})
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": reply}]}}]})

    return httpx.Client(transport=httpx.MockTransport(handler))


class TestWhichProvidersAppear:
    def test_no_keys_means_deterministic_only(self) -> None:
        setup = build_llm_chain(_settings(), store=InMemoryRateLimitStore())

        assert not setup.has_llm
        assert any("deterministic mapper only" in n for n in setup.notes)
        assert setup.chain.propose(INCIDENT, ["divert_arterial"])[1] == "deterministic"

    def test_only_the_configured_providers_are_built(self) -> None:
        groq_only = build_llm_chain(
            _settings(groq_api_key=GROQ_KEY), store=InMemoryRateLimitStore()
        )
        gemini_only = build_llm_chain(
            _settings(gemini_api_key=GEMINI_KEY), store=InMemoryRateLimitStore()
        )
        both = build_llm_chain(
            _settings(groq_api_key=GROQ_KEY, gemini_api_key=GEMINI_KEY),
            store=InMemoryRateLimitStore(),
        )

        assert groq_only.chain.provider_names == ["groq"]
        assert gemini_only.chain.provider_names == ["gemini"]
        assert both.chain.provider_names == ["groq", "gemini"]

    def test_a_configured_chain_really_calls_the_provider_with_its_key(self) -> None:
        seen: list[httpx.Request] = []
        setup = build_llm_chain(
            _settings(groq_api_key=GROQ_KEY, gemini_api_key=GEMINI_KEY),
            store=InMemoryRateLimitStore(),
            client=_answering_client(seen),
        )

        proposal, name = setup.chain.propose(INCIDENT, ["divert_arterial", "hold_current_plan"])

        assert name == "groq"  # the faster prior wins while both are fresh
        assert proposal.playbook_name == "divert_arterial"
        assert seen[0].headers["authorization"] == f"Bearer {GROQ_KEY}"

    def test_configured_models_and_limits_flow_through_to_the_notes(self) -> None:
        setup = build_llm_chain(
            _settings(
                groq_api_key=GROQ_KEY,
                groq_model="llama-test",
                groq_rpm=7,
                groq_rpd=99,
            ),
            store=InMemoryRateLimitStore(),
        )

        assert "llama-test (7/min, 99/day)" in " ".join(setup.notes)


class TestSecretsNeverLeak:
    def test_notes_contain_no_key(self) -> None:
        setup = build_llm_chain(
            _settings(groq_api_key=GROQ_KEY, gemini_api_key=GEMINI_KEY),
            store=InMemoryRateLimitStore(),
        )

        joined = " ".join(setup.notes)
        assert GROQ_KEY not in joined
        assert GEMINI_KEY not in joined

    def test_the_settings_repr_masks_keys(self) -> None:
        settings = _settings(groq_api_key=GROQ_KEY, gemini_api_key=GEMINI_KEY)

        assert GROQ_KEY not in repr(settings)
        assert GEMINI_KEY not in repr(settings)
        assert GROQ_KEY not in str(settings.model_dump())


class TestRedisHandling:
    def test_an_unreachable_redis_degrades_to_in_memory_and_says_so(self) -> None:
        setup = build_llm_chain(
            _settings(groq_api_key=GROQ_KEY, redis_url="redis://:s3cretpass@127.0.0.1:1/0")
        )

        joined = " ".join(setup.notes)
        assert "not reachable" in joined
        assert "s3cretpass" not in joined  # credentials in the URL are never echoed
        # ...and the chain still works, counting in memory.
        assert setup.chain.status()[0].headroom == 1.0


class TestSettings:
    def test_defaults(self) -> None:
        settings = _settings()

        assert settings.groq_model == "llama-3.1-8b-instant"
        assert (settings.groq_rpm, settings.groq_rpd) == (30, 14400)
        assert (settings.gemini_rpm, settings.gemini_rpd) == (10, 250)
        assert settings.redis_url == "redis://localhost:6379/0"

    def test_environment_variable_names(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GROQ_API_KEY", GROQ_KEY)
        monkeypatch.setenv("GOOGLE_API_KEY", GEMINI_KEY)  # the alias Google's own docs use
        monkeypatch.setenv("ATLAS_GEMINI_MODEL", "gemini-test")
        monkeypatch.setenv("ATLAS_GROQ_RPM", "12")
        monkeypatch.setenv("ATLAS_REDIS_URL", "redis://example:6380/2")

        settings = _settings()

        assert settings.groq_api_key is not None
        assert settings.groq_api_key.get_secret_value() == GROQ_KEY
        assert settings.gemini_api_key is not None
        assert settings.gemini_api_key.get_secret_value() == GEMINI_KEY
        assert settings.gemini_model == "gemini-test"
        assert settings.groq_rpm == 12
        assert settings.redis_url == "redis://example:6380/2"

    def test_non_positive_limits_are_rejected(self) -> None:
        with pytest.raises(ValueError):
            _settings(groq_rpm=0)
