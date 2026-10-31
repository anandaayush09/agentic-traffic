"""LLM-backed playbook proposals: real Groq/Gemini providers, Redis-shared rate limits,
and quota/health-aware routing between them. See each module's docstring."""

from atlas.cognitive.llm.factory import LlmSettings, LlmSetup, build_llm_chain
from atlas.cognitive.llm.limits import (
    InMemoryRateLimitStore,
    ProviderLimits,
    QuotaTracker,
    RateLimitStore,
    RedisRateLimitStore,
    ResilientRateLimitStore,
)
from atlas.cognitive.llm.providers import GeminiProvider, GroqProvider, parse_proposal
from atlas.cognitive.llm.router import AdaptiveProviderChain, ProviderStatus, RoutedProvider

__all__ = [
    "AdaptiveProviderChain",
    "GeminiProvider",
    "GroqProvider",
    "InMemoryRateLimitStore",
    "LlmSettings",
    "LlmSetup",
    "ProviderLimits",
    "ProviderStatus",
    "QuotaTracker",
    "RateLimitStore",
    "RedisRateLimitStore",
    "ResilientRateLimitStore",
    "RoutedProvider",
    "build_llm_chain",
    "parse_proposal",
]
