#!/usr/bin/env python
"""Live check of the configured LLM providers and the shared rate-limit store: makes ONE
real, tiny call to each provider that has a key and reports what happened, then one call
through the quota-aware router. This is the step that verifies the Groq/Gemini request
shapes against a real key (the unit tests only prove them against the documented shapes).

Keys come from the environment or a `.env` file in the repo root (never from arguments,
never printed):

    GROQ_API_KEY=...
    GEMINI_API_KEY=...
    ATLAS_REDIS_URL=redis://localhost:6379/0     # optional; in-memory if unreachable

Usage:
    python scripts/check_llm_providers.py

Exit code 0 if every configured provider answered with a valid proposal, 1 otherwise (or
if no provider is configured). Each provider call uses a few hundred tokens -- trivial next
to any free tier. The direct per-provider calls do not count against the shared per-minute
budget; the one routed call at the end does.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atlas.cognitive.llm import LlmSettings, build_llm_chain  # noqa: E402
from atlas.cognitive.playbooks import PlaybookRegistry  # noqa: E402
from atlas.cognitive.provider_chain import ProviderError  # noqa: E402

SAMPLE_INCIDENT: dict[str, object] = {
    "incident_id": "check-llm-1",
    "class": "accident",
    "status": "detected",
    "confidence": 0.9,
    "location": {"link_ids": ["42430342"]},
}


def main() -> int:
    registry = PlaybookRegistry()
    setup = build_llm_chain(LlmSettings(), registry=registry)
    print("=== Configuration ===")
    for note in setup.notes:
        print(" ", note)
    if not setup.has_llm:
        print("\nNothing to check: set GROQ_API_KEY and/or GEMINI_API_KEY (env or .env).")
        return 1

    print("\n=== Direct call to each provider ===")
    failures = 0
    for routed in setup.chain.providers:
        started = time.monotonic()
        try:
            proposal = routed.provider.propose(dict(SAMPLE_INCIDENT), registry.names())
        except ProviderError as exc:
            failures += 1
            print(f"  [FAIL] {routed.name:7s} {type(exc).__name__}: {exc}")
            continue
        elapsed_ms = (time.monotonic() - started) * 1000
        print(
            f"  [ OK ] {routed.name:7s} {elapsed_ms:6.0f} ms  -> {proposal.playbook_name} "
            f"{proposal.params}"
        )
        print(f"           rationale: {proposal.rationale}")

    print("\n=== One call through the quota-aware router ===")
    proposal, answered_by = setup.chain.propose(dict(SAMPLE_INCIDENT), registry.names())
    print(f"  answered by: {answered_by} -> {proposal.playbook_name}")

    print("\n=== Routing state (best candidate first) ===")
    print(f"  {'provider':9s} {'headroom':>8s} {'reliab.':>8s} {'latency':>9s} {'cooldown':>9s}")
    for s in setup.chain.status():
        print(
            f"  {s.name:9s} {s.headroom:8.0%} {s.reliability:8.0%} "
            f"{s.latency_ms:7.0f}ms {s.cooldown_s:8.0f}s {'' if s.available else '(skipped)'}"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
