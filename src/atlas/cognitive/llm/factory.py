"""Builds the LLM provider chain from the environment / `.env`.

    GROQ_API_KEY            Groq key (provider skipped if unset)
    GEMINI_API_KEY          Gemini key (GOOGLE_API_KEY also accepted; skipped if unset)
    ATLAS_GROQ_MODEL        default llama-3.1-8b-instant
    ATLAS_GEMINI_MODEL      default gemini-2.5-flash
    ATLAS_GROQ_RPM / _RPD   request caps to enforce client-side (default 30 / 14400)
    ATLAS_GEMINI_RPM / _RPD                                       (default 10 / 250)
    ATLAS_REDIS_URL         shared rate-limit state (default redis://localhost:6379/0)

Free-tier numbers differ by model and change over time, so the defaults are deliberately
conservative and every cap is overridable -- check your own account's limits page.
Keys are `SecretStr`, so printing or logging the settings object never leaks them. With no
keys at all the chain is simply the deterministic mapper, same as before this existed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import urlparse

import httpx
from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from atlas.cognitive.llm.limits import (
    InMemoryRateLimitStore,
    ProviderLimits,
    RateLimitStore,
    RedisRateLimitStore,
    ResilientRateLimitStore,
)
from atlas.cognitive.llm.providers import GeminiProvider, GroqProvider
from atlas.cognitive.llm.router import AdaptiveProviderChain, RoutedProvider
from atlas.cognitive.playbooks import PlaybookRegistry
from atlas.common.logging import get_logger

logger = get_logger(__name__)

# Rough typical latencies, used only to break ties before real measurements exist.
_GROQ_PRIOR_LATENCY_MS = 400.0
_GEMINI_PRIOR_LATENCY_MS = 1500.0


class LlmSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)

    groq_api_key: SecretStr | None = Field(default=None, validation_alias="GROQ_API_KEY")
    gemini_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_API_KEY")
    )
    groq_model: str = Field(default="llama-3.1-8b-instant", validation_alias="ATLAS_GROQ_MODEL")
    gemini_model: str = Field(default="gemini-2.5-flash", validation_alias="ATLAS_GEMINI_MODEL")
    groq_rpm: int = Field(default=30, gt=0, validation_alias="ATLAS_GROQ_RPM")
    groq_rpd: int = Field(default=14400, gt=0, validation_alias="ATLAS_GROQ_RPD")
    gemini_rpm: int = Field(default=10, gt=0, validation_alias="ATLAS_GEMINI_RPM")
    gemini_rpd: int = Field(default=250, gt=0, validation_alias="ATLAS_GEMINI_RPD")
    redis_url: str = Field(default="redis://localhost:6379/0", validation_alias="ATLAS_REDIS_URL")


@dataclass
class LlmSetup:
    chain: AdaptiveProviderChain
    notes: list[str] = field(default_factory=list)
    """Human-readable lines about what was configured (never contains a key)."""

    @property
    def has_llm(self) -> bool:
        return bool(self.chain.provider_names)


def _make_store(redis_url: str, notes: list[str]) -> RateLimitStore:
    fallback = InMemoryRateLimitStore()
    where = urlparse(redis_url)
    target = f"{where.hostname}:{where.port or 6379}"  # never echo credentials in the URL
    try:
        import redis

        client = redis.Redis.from_url(
            redis_url, socket_connect_timeout=1.0, socket_timeout=1.0, decode_responses=True
        )
    except Exception as exc:  # missing package or a malformed URL
        notes.append(f"Rate limits: in-memory only ({type(exc).__name__}: cannot use Redis)")
        return fallback

    try:
        client.ping()
        notes.append(f"Rate limits: shared via Redis at {target}")
    except Exception:
        notes.append(
            f"Rate limits: Redis at {target} is not reachable -- using in-memory counters "
            "(per process) until it comes up"
        )
    # Wrapped either way: the client connects lazily, so Redis starting later is picked up
    # automatically, and a mid-run outage degrades to in-memory instead of failing calls.
    return ResilientRateLimitStore(RedisRateLimitStore(client), fallback)


def build_llm_chain(
    settings: LlmSettings | None = None,
    registry: PlaybookRegistry | None = None,
    store: RateLimitStore | None = None,
    client: httpx.Client | None = None,
) -> LlmSetup:
    """The chain for whichever providers have a key, ranked at call time by quota and
    health. `store` and `client` are injectable for tests (and the live check script)."""
    cfg = settings or LlmSettings()
    registry = registry or PlaybookRegistry()
    notes: list[str] = []
    providers: list[RoutedProvider] = []

    if cfg.groq_api_key is not None:
        providers.append(
            RoutedProvider(
                "groq",
                GroqProvider(
                    cfg.groq_api_key.get_secret_value(), cfg.groq_model, registry, client=client
                ),
                ProviderLimits(cfg.groq_rpm, cfg.groq_rpd),
                prior_latency_ms=_GROQ_PRIOR_LATENCY_MS,
            )
        )
        notes.append(f"Groq: {cfg.groq_model} ({cfg.groq_rpm}/min, {cfg.groq_rpd}/day)")
    if cfg.gemini_api_key is not None:
        providers.append(
            RoutedProvider(
                "gemini",
                GeminiProvider(
                    cfg.gemini_api_key.get_secret_value(), cfg.gemini_model, registry, client=client
                ),
                ProviderLimits(cfg.gemini_rpm, cfg.gemini_rpd),
                prior_latency_ms=_GEMINI_PRIOR_LATENCY_MS,
            )
        )
        notes.append(f"Gemini: {cfg.gemini_model} ({cfg.gemini_rpm}/min, {cfg.gemini_rpd}/day)")

    if not providers:
        notes.append("LLM: no GROQ_API_KEY / GEMINI_API_KEY set -- deterministic mapper only")
        return LlmSetup(AdaptiveProviderChain([], store or InMemoryRateLimitStore()), notes)

    return LlmSetup(
        AdaptiveProviderChain(providers, store or _make_store(cfg.redis_url, notes)), notes
    )
