# ADR-0012: Static frozen-snapshot public demo instead of a live backend

**Status:** Accepted (Week 10 part 5, commit `6d47692` — decided during the build, not in
the original design doc)

## Context

Week 10's roadmap deliverable called for a publicly demoable dashboard. A fully faithful
public demo would mean a live Cognitive API backend, reachable from the public internet, with
a Postgres-backed telemetry pipeline (per ADR-0003, never built in this project) behind it —
real infrastructure cost and an external hosting account, neither of which this development
environment can provide on its own.

## Decision

When this tradeoff was raised explicitly during the build, the choice made was: **do not
build a live backend for the public demo.** Instead, `scripts/export_public_demo_snapshot.py`
runs the real Week 9 scripted incident suite (10 real incidents, hand-labeled with the
playbook an operator would pick) through the real Supervisor LangGraph against the real
manhattan_midtown corridor, and freezes the results into
`frontend/src/public_demo_snapshot.json`. `npm run build:demo` (`VITE_STATIC_DEMO=true`)
produces a separate, fully static frontend build with zero backend calls and no mutating UI
(Approve/Reject replaced with inert "Would require operator approval" text) — deployable to
any free static host.

## Alternatives considered

- **A Postgres-backed live telemetry pipeline + a hosted backend:** the original, fuller
  option — rejected for this build because it requires infrastructure (a durable telemetry
  store, a publicly reachable API, hosting credentials) that doesn't yet exist and is outside
  what a static-artifact deliverable needs to prove.

## Consequences

- The public demo's every result is real (a genuine Supervisor run against a genuine
  incident, not fixture data), but frozen at export time — it cannot show a *new* incident
  live, only replay the 10 scripted ones. This is a real, named scope reduction from "public
  demo" in the fullest sense, not a hidden one.
- Structurally, not just by configuration, the public build has "zero control endpoints, no
  bus, no LLM keys" (`docs/security/threat_model.md`'s framing) — there is no backend at all
  to attack, which is a genuine security simplification, not only a cost one.
- Verified for real before shipping: both the normal and `:demo` builds compile clean, and
  the static build was served and inspected via browser network-request logging to confirm
  zero backend calls and correct read-only rendering of all 10 real incidents.
- The `dist/` output is a plain static site; deploying it publicly (choosing and paying for a
  host, if any) is left to whoever owns the repository, not automated by this project.
