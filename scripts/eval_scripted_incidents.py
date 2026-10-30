#!/usr/bin/env python
"""Runs the Week 9 scripted incident suite end to end and reports the roadmap's own
metrics (docs/design/13_roadmap.md: "scripted incident suite v1 (10 incidents) scoring
playbook accuracy + grounding rate"; docs/design/01_problem_definition.md's target:
"cognitive-plane quality: >= 80% correct playbook selection").

Uses a real `ProviderChain` with NO providers configured -- there are no LLM API keys
in this environment (same honest stance as `provider_chain.py`), so every incident here
exercises the terminal deterministic-mapper fallback for real. This IS the "demo runs
on deterministic playbooks... system value survives by design" mitigation
(docs/design/12_risks.md R4) measured across the whole suite, not just asserted.

Usage:
    python scripts/eval_scripted_incidents.py          # deterministic mapper only
    python scripts/eval_scripted_incidents.py --llm    # Groq/Gemini per GROQ_API_KEY /
                                                       # GEMINI_API_KEY (falls back per
                                                       # incident, so a scored row may
                                                       # still say provider=deterministic)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atlas.cognitive.incident_suite import run_suite  # noqa: E402
from atlas.cognitive.playbooks import PlaybookRegistry  # noqa: E402
from atlas.cognitive.provider_chain import ProposalSource, ProviderChain  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", action="store_true", help="use the configured LLM providers")
    args = parser.parse_args()

    registry = PlaybookRegistry()
    chain: ProposalSource
    label = "deterministic-mapper fallback"
    if args.llm:
        from atlas.cognitive.llm import build_llm_chain

        setup = build_llm_chain(registry=registry)
        for note in setup.notes:
            print(note)
        chain = setup.chain
        label = "LLM providers, deterministic fallback" if setup.has_llm else label
    else:
        chain = ProviderChain(providers=[])  # no LLM providers -- terminal fallback only

    summary = run_suite(chain, registry)

    print(f"\n=== Scripted incident suite (10 incidents, {label}) ===")
    for score in summary.scores:
        status = "OK" if score.playbook_correct else "MISS"
        grounded = "grounded" if score.grounded else "UNGROUNDED"
        valid = "valid" if score.params_valid else "INVALID"
        print(
            f"[{status:4s}] {score.incident_id:22s} expected={score.expected_playbook:22s} "
            f"actual={score.actual_playbook:22s} provider={score.provider_used:12s} "
            f"{grounded:10s} {valid}"
        )

    print(f"\nplaybook_accuracy: {summary.playbook_accuracy:.0%} ({len(summary.scores)} incidents)")
    print(f"grounding_rate:    {summary.grounding_rate:.0%}")
    print(f"param_sanity_rate: {summary.param_sanity_rate:.0%}")
    target = 0.80
    verdict = "MEETS" if summary.playbook_accuracy >= target else "BELOW"
    print(f"target (docs/design/01_problem_definition.md): >= {target:.0%} -- {verdict}")


if __name__ == "__main__":
    main()
