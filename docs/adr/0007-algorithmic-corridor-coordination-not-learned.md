# ADR-0007: Algorithmic (not learned) corridor coordination

**Status:** Accepted (design Part 4 §4.2; built Week 7 parts 1–2, commits `fd4291a`,
`048fcbb`)

## Context

Per-junction MARL alone struggles with longer-range coordination (e.g., timing consecutive
junctions along an arterial to pass a platoon of vehicles through on green — a "green wave").
The system needs some mechanism above the intersection level to supply this signal.

## Decision

A separate `CorridorCoordinator` service computes green-wave offsets algorithmically —
classic bandwidth-maximization from measured platoon speeds, plus a spillback detector
(exit-link occupancy over a threshold) — and publishes the result as an advisory
`corridor.hints` feature vector that member Intersection Agents consume as an *additional
observation input*, never as a command. Intersections remain the single writer of their own
signals (the bus architecture's single-writer principle, ADR-0002).

## Alternatives considered

- **Learned inter-agent communication** (e.g., graph-attention message passing between
  agents, à la CoLight): rejected for this build — coordination via learned communication is
  a research project in its own right, and an algorithmic hint keeps the system explainable
  and gives the RL layer a stable curriculum feature rather than a moving target. Documented
  as an explicit stretch goal: replace hints with learned GAT messages and A/B the two.

## Consequences

- Because hints are just another observation feature (default 0.0 when absent), the policy
  can be trained with hint-dropout and degrade smoothly if the coordinator is unavailable —
  the same graceful-degradation philosophy the control plane applies everywhere else.
- The coordinator deliberately owns a plain `SumoTwinEnv`, not the same `MultiAgentSignalEnv`
  the Intersection Agents drive, specifically because the latter inherently commands signals
  via the shield on every step — a read-only coordinator sharing that env type would become a
  second writer of junctions it should only be observing.
- Named, honest gap found and documented while building this (Week 7 part 2): the
  coordinator currently reads its own separate simulation instance rather than the one under
  live agent control, because the cross-process Redis state-sharing pipeline (ADR-0003) was
  never wired up — the same limitation noted there, surfaced concretely here.
