# ADR-0003: State tiering — Redis for hot state, Postgres+Timescale for durable state

**Status:** Accepted (design Part 3, §3.5)

## Context

ATLAS has two very different state-access patterns: the fast control loop needs the *latest*
per-intersection observation/signal-state/heartbeat with sub-millisecond reads and trivial
staleness detection (a watchdog checking key age); evaluation, audit, and the dashboard's
historical charts need durable, queryable, time-partitioned history of every bus event and
decision.

## Decision

Two stores, one per access pattern:

- **Redis** — latest per-intersection observation vectors, signal states, agent-health
  heartbeats, preemption locks. TTL'd keys make staleness detection a simple key-age check
  (the Watchdog pattern, ADR-0002's failure-recovery story).
- **PostgreSQL + TimescaleDB** — all bus events as hypertables partitioned by time, plus an
  append-only audit table (decision input hash, action, shield delta, policy version,
  explanation id) for the auditability NFR.

## Alternatives considered

- **Kafka alone, no Redis:** rejected — reading "latest value by key" from a log forces every
  reader to maintain its own consumer offset/compaction state for data it only ever needs the
  newest value of; Redis's native latest-value + TTL semantics are exactly this access
  pattern.
- **A dedicated time-series database (InfluxDB) instead of Timescale:** rejected — one
  relational engine with a time-series extension means fewer moving parts for a solo
  engineer, plain SQL for evaluation queries (Week 11's statistics work reads tripinfo/metric
  data this way), and better hiring-manager legibility than a specialized TSDB.

## Consequences

- Fast-loop code never touches SQL; durable-store code never sits on the control loop's
  critical path — the two concerns can be developed, tested, and reasoned about
  independently.
- Named, honest gap (documented in `docs/security/threat_model.md` and the Week 7/10 dev
  notes): the Postgres audit sink and the Redis-backed cross-process state pipeline were
  never wired up in this build — `IntersectionAgentService` and `CorridorCoordinatorService`
  each currently run their own in-process simulation state rather than sharing state through
  Redis as the design specifies. This is a real limitation of the as-built system, not
  silently assumed complete.
