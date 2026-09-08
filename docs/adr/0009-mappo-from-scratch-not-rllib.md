# ADR-0009: MAPPO implemented from scratch instead of RLlib

**Status:** Accepted (design Part 6 §6.1; built Week 5, commits `cf114c2`, `aaed0a0`,
`3eeee24`, `08609dd`, `d5a4fc9`)

## Context

The per-junction control policy needs a multi-agent RL algorithm; RLlib (or a similar
framework) would get a working policy running faster than writing the algorithm by hand.

## Decision

MAPPO (Yu et al. 2022's CTDE formulation — shared-parameter actor, per-agent-conditioned
centralized critic) implemented from scratch, CleanRL-style (~500 lines: network, rollout
buffer + GAE, clipped-surrogate PPO update), on top of Ray Core (not RLlib) for rollout
parallelism.

## Alternatives considered

- **RLlib** (or another off-the-shelf MARL framework): rejected — the explicit reasoning is
  learning value (the author is new to RL; owning the training loop is the point),
  debuggability, interview depth, and avoiding RLlib's API churn across versions.
- **Off-policy methods (QMIX, MADDPG):** rejected — off-policy sample efficiency matters less
  when samples come from a fast local simulator rather than a costly real-world rollout, and
  QMIX's monotonic value-factorization constraint adds fragility judged not worth taking on
  for a first RL implementation.
- **Single shared critic, not per-agent-conditioned:** an early implementation choice within
  the from-scratch build, caught and corrected before other code depended on it (commit
  `aaed0a0`, same session as the network's first commit): the critic originally produced one
  shared value per timestep, but Part 6.1's rewards are per-agent, leaving nothing consistent
  to regress to. Fixed to match the actual MAPPO paper — a one-hot agent index appended to
  the joint observation, one shared network still producing per-agent values.

## Consequences

- Every core algorithmic piece was hand-verified against known-correct references before
  being trusted: the rollout buffer's GAE recursion against hand-computed two-agent values to
  4 decimal places (confirming no cross-agent leakage), and PPO's clip asymmetry against a
  hand-derived example (positive advantage + large ratio → clipped; negative advantage +
  large ratio → not clipped) before writing the implementation.
- Owning the training loop directly surfaced and let the project fix two real training bugs
  a framework might have hidden or handled silently: an import-ordering native-library
  collision (`mlflow` before `libsumo` on Windows) and a reward-scale instability where an
  under-normalized raw max-pressure reward made a trained policy perform *worse* than random
  until a `RunningRewardNormalizer` was added — caught by actually comparing against a random
  policy baseline, not by assuming the loop ran correctly because it ran without error.
- Real cost of this decision, stated honestly: as of Week 11, no MAPPO checkpoint has been
  trained to real convergence in this development environment — Week 5/6's runs are
  documented short verification runs (proving the loop trains at all, and beats a random
  baseline by ~66% on mean pressure over a fixed-seed sanity check), not the multi-hour-to-
  multi-day Colab/Kaggle convergence run the design's own training-throughput math calls for.
  See ADR-0013 for the full, honest accounting of what this means for Weeks 11–12's
  evaluation scope.
