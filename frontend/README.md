# ATLAS Operations Dashboard

React + TypeScript + Vite + MapLibre GL dashboard (docs/design/13_roadmap.md's Week
10 deliverable): map view, approval queue, and XAI panel against the real Cognitive
API (`src/atlas/services/cognitive_api.py`).

## Live/dev mode

```bash
npm install
npm run dev
```

Then run a real backend (see `../scripts/run_dashboard_demo_api.py`) and paste the
operator JWT it prints into the dashboard's "Operator token" field to approve/reject
incidents or submit a demo one.

`npm run build` produces the same live-backed app as a static bundle (`dist/`) --
still needs a real Cognitive API reachable from wherever it's hosted.

## Public read-only demo

```bash
npm run build:demo
```

Builds a **separate, fully static** variant (`VITE_STATIC_DEMO=true`, set by
`.env.demo`) with:
- No backend calls at all -- incidents come from `src/public_demo_snapshot.json`, a
  real, frozen snapshot of the Week 9 scripted incident suite run through the real
  Supervisor graph against the real manhattan_midtown corridor
  (`../scripts/export_public_demo_snapshot.py` regenerates it).
- No Approve/Reject buttons or operator-token/API-base-URL inputs -- a pending
  incident shows "Would require operator approval" as inert text instead.

This matches docs/design/10_deployment.md's public demo constraint ("zero control
endpoints, no bus, no LLM keys") literally: there is no backend to call, so there is
nothing to secure or misuse. The resulting `dist/` is a plain static site -- deploy it
to any static host (GitHub Pages, Netlify, Vercel, Cloudflare Pages) with no server
component needed.
