# ADR-0013: R1/R2/R4 risk-register outcomes, as actually realized

**Status:** Accepted (Week 11/12 — recording what actually happened, against
[12_risks.md](../design/12_risks.md)'s pre-committed descope plan)

## Context

The design's risk register (Part 12) pre-committed specific descope actions for three
technical risks, precisely so that hitting them mid-project would be a planned configuration
change, not a crisis. This ADR records, honestly, what actually happened for each — because
the real outcome differs in an important way from each risk's literal framing, and a
portfolio project that claims rigor should say so precisely rather than gloss over the
difference.

## R1 — "MARL doesn't converge at district scale"

**Design's framing:** training is attempted at district scale and found to underperform
max-pressure at the Week 7 go/no-go checkpoint; descope to corridor-scale RL (5–10 junctions)
+ max-pressure elsewhere.

**What actually happened:** the go/no-go comparison itself never occurred, because no
district-scale MAPPO training run was ever executed to convergence in this development
environment. Week 5/6's training runs (`scripts/train_mappo_corridor.py`,
`train_mappo_district.py`) are documented short verification runs proving the training loop
works correctly (a trained corridor-stage policy beat a random baseline by ~66% on mean
pressure, single-seed sanity check) — not the multi-hour-to-multi-day Colab/Kaggle session
the design's own throughput math (§6.1: "5–15M steps per curriculum stage... an
overnight-to-weekend session") calls for. This is a **compute-availability gap**, not a
disproven hypothesis about MARL's ability to converge at district scale.

**Realized outcome:** effectively the same shipped configuration R1's descope prescribes —
evaluation and the production baseline set (Week 11) use max-pressure, fixed-time, and
SUMO-actuated as the three real, statistically evaluated controllers; `mode="rl"` is
explicitly out of scope in `atlas.evaluation.runner` and fails loudly rather than silently
using an under-trained checkpoint. No corridor-scale RL checkpoint has been evaluated
statistically either — it exists only as an unevaluated Week 5 training artifact.

## R2 — "Colab/Kaggle throughput insufficient"

**Design's framing:** measured steps/hour on Colab/Kaggle falls short; descope to
AM-peak-only episodes and fewer eval-suite variants, but never fewer seeds.

**What actually happened:** Colab/Kaggle were never used in this development session at all
— all training and evaluation ran locally on Windows CPU (`backend="traci"`, forced by the
libsumo/Ray DLL collision documented in ADR-0004). Measured throughput (Week 4): 68,392
steps/hour on 1 worker, 152,571 steps/hour on 8 workers — a small fraction of the design's own
"4 Kaggle vCPUs × parallel envs ≈ 1–2M env-steps/hour" estimate, and explicitly labeled a
pessimistic lower bound, not evidence against Kaggle's real ceiling.

**Realized outcome:** R2's prescribed rule was followed in spirit even though the specific
trigger (a Colab/Kaggle measurement) never fired: Week 11's evaluation pilot reduced scenario
*conditions* (5, now 9 after the Week 12 close-out variants — see
`configs/scenarios/manhattan_midtown.yaml` — against the design's ~35-cell grid) while
keeping the full `NUM_SEEDS=10` the design specifies, after an initial 5-seed pilot run
demonstrated the exact statistical-power problem R2 is trying to prevent (large effect sizes
that didn't survive Holm-Bonferroni correction at n=5). Scope was cut; rigor was not.

## R4 — "Free-tier LLM limits or provider changes break the cognitive plane"

**Design's framing:** a free-tier LLM provider's limits are hit or its terms change mid-
project; descope to "deterministic playbooks + Ollama," with the cross-provider LLM
comparison table shrinking.

**What actually happened:** no LLM provider was ever wired up in this build — not Groq, not
Gemini, and not the local-first Ollama fallback the descope plan itself names. The
`ProviderChain`'s fallback *logic* is fully built and tested (ADR-0010), but every real
provider client is unimplemented, because there is no way to verify a real provider's API
surface without an actual key and a live call, and no key exists in this environment.

**Realized outcome:** a fuller descope than R4 anticipates. The design's fallback plan
assumes at least a local Ollama model stays in the loop; the actual system runs entirely on
the deterministic mapper. Week 9's scripted-incident evaluation
(`scripts/eval_scripted_incidents.py`) measured the mapper alone at
playbook_accuracy=100%/grounding_rate=100%/param_sanity_rate=100% (10/10) — a real, strong
result, but one that validates the mapper's own curated defaults against operator
expectations, not LLM reasoning quality. The "LLM comparison table" R4 says should shrink
does not yet exist at all; adding one real provider (even local Ollama) is future work, not
completed and reduced.

## Consequences

- Every evaluation claim made in `docs/report/evaluation_pilot_report.md` and elsewhere in
  this repository is scoped to the three non-RL, non-LLM-reasoning controllers/mechanisms
  that were actually built and verified — RL and LLM-reasoning claims are consistently
  absent, not weakened-but-present.
- The honest path to closing R1 and R4 is unchanged by anything a future session working
  inside this same environment can do alone: R1 needs real GPU/Colab-Kaggle compute time; R4
  needs either a free-tier API key or a local Ollama installation, neither available here.
  Both are one bounded, well-scoped piece of future work each — the harness, evaluation
  protocol, and provider-chain fallback logic that would consume them are already built and
  tested.
