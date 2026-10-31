# ADR-0014: Quota- and health-aware LLM routing with Redis-shared rate limits

**Status:** Accepted (builds on [ADR-0010](0010-llm-provider-chain-with-deterministic-terminal-fallback.md);
Phase 2 of the post-Week-12 plan)

## Context

ADR-0010 built the fallback *logic* (ordered providers, token buckets, a deterministic
terminal) against fake providers, and deliberately deferred real Groq/Gemini clients until a
real key could verify their request shapes. Two real free-tier providers raise questions a
fixed order and a per-process token bucket do not answer:

- Both enforce **per-minute and per-day request caps**, and the numbers differ by model and
  change over time. More than one process (the API, evaluation scripts) can draw on the same
  key, so the counters must be shared and atomic.
- "Groq first, then Gemini" wastes a healthy provider while the first one is rate-limited,
  failing, or out of daily budget. The useful question per incident is *which provider
  should answer now?*

## Decision

`atlas.cognitive.llm`:

1. **`RateLimitStore`** (swap point, same pattern as `LockStore`): `RedisRateLimitStore`
   (shared) and `InMemoryRateLimitStore`, wrapped by `ResilientRateLimitStore`, which falls
   back to in-memory counters when Redis errors, so a Redis outage degrades rate-limit
   accuracy and never incident handling. Counters are fixed time buckets (minute, UTC day);
   a request is *reserved* (increment, refund if over the cap) before the call so concurrent
   processes cannot all take the last slot.
2. **`AdaptiveProviderChain`** ranks providers per incident by `score = headroom x
   reliability` (unused fraction of the tighter of the minute/day budgets; moving-average
   success rate), breaking near-ties (within 0.05) by observed latency. Providers on
   **cooldown** (a 429, honouring `Retry-After` / Gemini's `retryDelay`; a rejected key for
   an hour; other errors for 10 s) or out of budget are skipped. It ends, as before, at the
   deterministic mapper and never raises. Health stats start optimistic and expire after ten
   idle minutes, so a provider that failed is retried and can recover instead of being
   starved by its own record.
3. **`GroqProvider` / `GeminiProvider`** are plain `httpx` REST calls, not vendor SDKs: two
   small endpoints don't justify two dependency trees on a project that has repeatedly paid
   for native-binary surface on Windows.
4. **Free-tier limits are configuration** (`ATLAS_GROQ_RPM` etc.), with conservative
   defaults, because the published numbers disagree between sources and change.

## The LLM proposes; it does not decide

A model reply is accepted only if it is one JSON object naming a playbook from the registry
whose params validate against that playbook's own model, every location it names is one the
incident actually reported, and durations are sane. Anything else is a provider failure and
the chain moves on. Only an allow-listed subset of incident fields reaches the model, and the
prompt tells it the incident is data. The what-if simulation then verifies the proposal
(ADR-0006), and disruptive playbooks still wait for a human (ADR-0005's spirit): a
hallucinating or prompt-injected model cannot reach the road.

## Consequences

- With no keys the system behaves exactly as before (deterministic mapper).
- Verified: HTTP behaviour and every failure mode against a mocked transport; the Redis store
  against `fakeredis`; a Groq-shaped proposal through the real Supervisor and a real SUMO
  what-if. **Not verified here:** the providers' real response behaviour -- that needs a key
  and is what `scripts/check_llm_providers.py` is for.
- Fixed-window counters allow a brief burst at a window boundary (up to 2x the per-minute
  cap across the edge). Acceptable against free-tier limits that the provider also enforces,
  and a 429 simply triggers a cooldown.
- Whether an LLM's proposals are *better* than the deterministic mapper's is an empirical
  question: `python scripts/eval_scripted_incidents.py --llm` scores both on the same
  10-incident suite (accuracy, grounding, parameter sanity).
