# ADR-0010: LLM provider chain with a deterministic terminal fallback

**Status:** Accepted (design Part 4 §4.4, risk R4; built Week 9 parts 3–4, commits
`bee4719`, `846c65d`)

## Context

The cognitive plane's reasoning is meant to come from an LLM, but the project's explicit
budget constraint is **$0** — free-tier providers (Groq, Gemini) have hard rate limits and
can change terms or availability at any time (risk R4). A design that goes silent or errors
out whenever a free-tier quota is hit, or a provider changes its terms, is not acceptable for
a system whose core selling point is continuous incident response.

## Decision

A `ProviderChain`: an ordered list of providers, each guarded by a client-side `TokenBucket`
rate limiter, falling through to the next on failure or exhaustion — terminating, always, in
a **deterministic mapper** (`map_incident_to_playbook`, a hand-written incident-class →
default-playbook lookup covering all 10 incident classes) that never raises and always
produces a usable result.

## Alternatives considered

- **Fail closed when the LLM chain is exhausted** (surface an error, require operator
  manual intervention): rejected — this would mean the system's value *disappears* under
  exactly the free-tier conditions the project is designed to run under, rather than
  *degrading gracefully*.

## Consequences

- The system's value degrades from "reasoned" (LLM-parameterized playbook selection) to
  "scripted" (fixed per-class default) under provider pressure, but never to "absent" — the
  same graceful-degradation philosophy the control plane applies with its shield/fallback
  ladder (ADR-0005), applied here to the cognitive plane.
- Verified for real, not just unit-tested against fakes: Week 9's scripted incident suite
  (`scripts/eval_scripted_incidents.py`) ran the real `ProviderChain` with zero providers
  configured (no LLM keys exist in this development environment) — every one of 10 incidents
  exercised the terminal deterministic-mapper fallback for real, achieving
  playbook_accuracy=100%, grounding_rate=100%, param_sanity_rate=100%, meeting the design's
  ≥80% target. Honest reading, stated in the dev notes at the time: this validates that the
  deterministic mapper's fixed defaults match operator expectations — a real but expected
  result given it's a curated 1:1 mapping — not LLM judgment quality, which remains untested
  because it is blocked on real provider API keys this environment doesn't have.
- Real Groq/Gemini/Ollama client classes were deliberately never implemented in this build:
  unlike, say, the YOLO perception pipeline (Week 8), there is no way to verify a real LLM
  provider's API surface without an actual key and a live call. The `ProviderChain`'s
  fallback *logic* is fully built and tested against fakes; wiring in one real provider class
  later is a small, isolated addition (one `propose` method per provider), not a redesign.
