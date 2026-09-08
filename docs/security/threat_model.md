# Threat Model

Week 10 deliverable (docs/design/13_roadmap.md: "threat-model doc";
docs/design/00_executive_summary.md: "threat model, JWT-protected control APIs, and a
read-only public demo"; docs/design/01_problem_definition.md's NFR table: "JWT on all
mutating APIs; schema validation on all bus messages; public demo strictly read-only;
threat model documented").

**Scope and honesty note, up front**: this documents the system as it is actually
built through Week 10, not the full frozen architecture's eventual scope. Where a
mitigation the design calls for doesn't exist yet (Redis, Postgres audit store, TLS
on the bus), that's stated explicitly as a gap, not silently assumed solved --
matching this project's discipline everywhere else (Week 5's "correct but
unverified" Docker note, Week 9's honest anomaly-detection false-positive rates).

## 1. Assets

| Asset | Why it matters |
|---|---|
| Signal control (`POST /mode` on Intersection Agent) | Wrong value = real (simulated) traffic disruption; the literal "physical" consequence in this system |
| Playbook approval (`POST /incidents/{id}/approve\|reject`, `POST /incidents`) | Approving triggers a real control-plane action via the Supervisor graph |
| Operator JWTs | Possession = ability to do both of the above |
| Incident/decision state (in-memory today) | Confidentiality is low-stakes (no PII); integrity matters -- a falsified incident could trigger a bogus playbook |
| The digital twin itself (SUMO process) | Not a real intersection -- a compromised twin can't hurt anyone, but its correctness matters for portfolio/evaluation validity |

There is no PII anywhere in this system: incidents, telemetry, and decisions are all
about intersections and vehicles in a simulation, never people. That materially
shrinks the blast radius of every threat below.

## 2. Trust boundaries (as built, not as designed)

```
Operator (browser, frontend/)
        | HTTPS in a real deploy; HTTP locally (kind/compose)
        v
Cognitive API (src/atlas/services/cognitive_api.py)   Intersection Agent (.../intersection_agent.py)
  - GET /incidents, GET /incidents/{id}: unauthenticated  - GET /health, GET /mode: unauthenticated
  - POST /incidents, /approve, /reject: JWT (require_operator)  - POST /mode: JWT (require_operator)
        |
        v
Supervisor LangGraph (in-process) -> ProviderChain (no real LLM keys configured
anywhere in this repo) -> deterministic mapper (always available, never calls out)
        |
        v
SUMO twin (libsumo, in-process) -- no network exposure of its own
```

Watchdog and Corridor Coordinator (`.../watchdog.py`, `.../corridor_coordinator.py`)
expose only `GET` endpoints (health/status/corridor-state) -- no JWT needed on those
today because there is nothing mutating to protect, matching
docs/design/03_system_architecture.md section 3.6's "Read-only APIs... are
unauthenticated."

## 3. Threats considered

### 3.1 Unauthenticated control action (the headline risk)

**Threat**: anyone who can reach a service's HTTP port issues a mutating request
directly (bypassing the dashboard) and changes a live intersection's mode, or
approves/injects a playbook.

**Status**: mitigated for the two mutating surfaces that exist today.
`atlas.services.auth.require_operator` (HS256 JWT, shared secret) gates:
- Cognitive API: `POST /incidents`, `POST /incidents/{id}/approve`,
  `POST /incidents/{id}/reject`
- Intersection Agent: `POST /mode`

Verified for real, not just asserted: `tests/unit/test_auth.py` (missing header,
garbage token, wrong signing secret, expired token all rejected with 401);
`tests/unit/test_cognitive_api.py`'s `TestHttpLayer` (401 without a token, 409 on
re-resolving an already-decided incident); `tests/unit/
test_intersection_agent_service.py::test_set_mode_without_a_token_is_rejected` (this
endpoint had NO auth at all through Week 9 -- the module's own docstring flagged it
as deferred to this exact week, closed here, not silently left).

**Known gap**: `POST /incidents` being JWT-gated is a deliberate choice beyond what
docs/design/03_system_architecture.md section 3.6 literally lists (only
`/playbooks/{id}/approve|reject` and `/incidents/{id}` are named) -- included because
submitting an incident can itself trigger a real AUTO_APPLY action
(docs/design/04_agent_architecture.md section 4.4), the same class of consequence as
approve/reject. In the frozen design, incidents arrive via the `incident.events.v1`
bus topic from the anomaly-detection service, not this HTTP endpoint at all --
`POST /incidents` exists only because no live Kafka/Redpanda runs locally (documented
since Week 5), so it's this service's own substitute ingestion path. **A real bus
consumer replacing it should NOT require operator JWT** -- an automated detector is
not "the operator"; it needs its own service-identity auth (e.g. mTLS between
services, or a bus-level ACL), not this human-operator token. This is a real,
intentional scoping gap for a future iteration, not an oversight.

### 3.2 JWT secret handling

**Threat**: the signing secret leaks, letting an attacker mint arbitrary operator
tokens.

**Status, honestly split**:
- **Local dev / kind demo**: `local-demo-secret-do-not-use-in-prod-32-bytes`, the
  same value in `deploy/helm/atlas/values.yaml`'s default,
  `scripts/run_dashboard_demo_api.py`'s fallback, and every test file. This is
  intentional and safe *only* because nothing behind it is real -- the name says so,
  and it must never be reused anywhere a real consequence exists.
- **kind deployment**: injected via a real Kubernetes `Secret`
  (`deploy/helm/atlas/templates/secret-cognitive-api.yaml`), not baked into the image
  or committed in plaintext outside `values.yaml`'s own dev default -- but
  `values.yaml`'s default IS the demo secret, so a real deployment MUST override it
  (`--set cognitiveApi.jwtSecret=<real value>` or a separate values file never
  committed). **Gap**: nothing currently enforces that override -- Helm will happily
  install with the demo secret in any environment. A production chart should refuse
  to install without an explicit override (e.g. `required` in the template, or no
  default at all).
- **Rotation**: not implemented. A leaked secret today requires a redeploy with a new
  value; every existing token becomes invalid at once (no grace period / key
  rotation scheme). Acceptable for a portfolio project's current scope, named here
  as a real gap rather than ignored.

### 3.3 Cross-origin requests (CORS)

**Threat**: a malicious page in the operator's browser makes authenticated requests
to the Cognitive API using a token the browser already holds (CSRF-style), or an
unrelated origin reads response data it shouldn't.

**Status**: `create_app`'s CORS middleware (`src/atlas/services/cognitive_api.py`)
allows only an explicit origin list (`http://localhost:5173` by default, the Vite
dev server), never `*`. Since the JWT is sent as an `Authorization: Bearer` header
(not a cookie), classic CSRF -- which relies on the browser auto-attaching cookies --
doesn't apply here: a malicious page cannot forge the header without already having
the token, and if it already has the token, CORS is not the control that would have
stopped it. CORS's actual job here is narrower and correctly scoped: preventing an
unrelated origin's script from reading Cognitive API responses in a victim's browser.

### 3.4 Prompt injection / LLM provider compromise

**Threat**: an LLM provider in the chain (Groq/Gemini/Ollama) is manipulated via
crafted incident text into proposing a dangerous playbook, or leaks its API key.

**Status**: **not applicable today** -- `ProviderChain` (`src/atlas/cognitive/
provider_chain.py`) has zero real LLM providers wired in anywhere in this repo (no
API keys exist in this environment, documented since Week 9 Part 4); every incident
in this system resolves via the deterministic mapper, a fixed lookup table with no
free-form text interpretation at all. This threat is real for the *frozen
architecture's eventual scope*, not for what's running today, and the existing
design already bounds it structurally when a real provider IS wired in: (1) every
playbook is a typed pydantic model (`src/atlas/cognitive/playbooks.py`) -- an LLM can
select and parameterize a playbook, never invent a new kind of action (docs/design/
04_agent_architecture.md section 4.4: "bounds the blast radius of model error"); (2)
`switch_to_max_pressure` and `hold_current_plan` are the only auto-applying
playbooks, and only when a real what-if simulation (`run_what_if`) confirms
improvement -- every other proposal always pauses for human approval
(`supervisor.py`'s `decide` node), so a manipulated LLM proposal for e.g.
`divert_arterial` can suggest nonsense but cannot execute it unattended.

### 3.5 Data at rest / audit trail

**Threat**: no record of who approved what, or a compromised process silently
altering decision history.

**Status**: **not built yet**. `docs/design/03_system_architecture.md` section 3.5
calls for an append-only Postgres audit log (decision log: input snapshot hash,
action, shield delta, policy version, explanation id); today, `CognitiveApiService`
and `DecisionLogStore` (`src/atlas/cognitive/analyst.py`) are both in-memory,
per-process, and lost on restart. This is a real, named gap -- not a security
control that was implemented and is merely undocumented. A real deployment needs
this before "audit trail" can be claimed at all.

### 3.6 Bus security (Kafka/Redpanda)

**Threat**: anyone on the network publishes forged `control.commands.v1` /
`incident.events.v1` messages, or reads sensitive telemetry off the bus.

**Status**: **not addressed**. `docker-compose.yml`'s Redpanda service runs with no
authentication, no ACLs, and no TLS (`PLAINTEXT://` listeners) -- acceptable only
because there is no PII on the bus (section 1) and because this compose stack is
explicitly a local dev/demo environment, never internet-exposed. Every consumed
message IS validated against its JSON Schema (`schemas/`, enforced since Week 5's
`SchemaRegistry`) before being acted on, which stops a malformed-but-not-malicious
message from crashing a consumer, but schema validity is not authentication -- a
schema-valid forged message from an unauthorized publisher would still be accepted.
A real deployment needs SASL/mTLS on the Kafka listeners and topic-level ACLs before
the bus should ever leave a private network.

## 4. Public demo (docs/design/10_deployment.md's `public demo` environment)

The frozen design's own constraint -- **"zero control endpoints, no bus, no LLM
keys"** -- is the single most load-bearing mitigation in this whole threat model, and
it is structural, not a configuration flag someone could forget: the public demo
environment is a *different deployment* (frontend + a read-only query API + a
Postgres snapshot, per section 10.1), not this same Cognitive API with auth stripped
out. There is no mutating endpoint to secure because none is deployed. This sidesteps
essentially every threat in section 3 by removing the asset, which is a stronger
guarantee than any auth scheme layered on top of a live control surface would be.

## 5. Summary table

| Threat | Mitigation | Status |
|---|---|---|
| Unauthenticated control action | JWT (`require_operator`) on every mutating endpoint | **Done**, verified with real 401/409 tests |
| JWT secret leak | k8s Secret, not baked into image | **Partial** -- default value is a known demo secret; nothing enforces overriding it |
| CSRF via forged cross-origin request | Bearer token (not cookie) + explicit CORS origin allowlist | **Done** |
| Malicious/manipulated LLM proposal | Typed playbook params + what-if-gated auto-apply + human approval fallback | **Structurally ready; not yet exercisable** (no real LLM provider wired in) |
| Tampered/lost audit trail | Postgres append-only audit log | **Not built** -- in-memory only today |
| Forged/eavesdropped bus messages | Schema validation (done) + SASL/mTLS/ACLs (not done) | **Partial** |
| Public demo exposure | Separate read-only deployment, no control surface at all | **Structural, by design** |

## 6. What a production deployment would still need

In priority order: (1) enforce a non-default JWT secret at install time (refuse to
deploy with the demo value); (2) the Postgres audit log, before any real "auditable"
claim; (3) SASL/TLS on Redpanda once the bus carries anything from a network with
untrusted actors on it; (4) JWT rotation / short-lived tokens with refresh, rather
than today's single long-lived shared secret; (5) rate limiting on the Cognitive
API's unauthenticated read endpoints (currently open to anyone who can reach the
port, which is acceptable for kind/local dev but not for anything internet-facing).
