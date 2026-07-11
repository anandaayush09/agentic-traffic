# Part 14 — Multi-Persona Design Review (frozen 2026-07-11)

This review was performed against the original, more expansive design (RLlib, CrewAI+
LangGraph both present, 12 always-on agent processes including standalone Weather/
Construction/Police/VIP agents, city-scale 100+ intersection ambitions, a real cloud K8s
target, hand-rolled consensus for agent coordination, METR-LA as the primary forecasting
deliverable). Findings below are organized by persona; **every finding produced a concrete
change**, already incorporated into Parts 1–13. Nothing here is aspirational — this is the
record of what changed and why. Per the user's instruction, **the architecture in Parts
1–13 is now frozen; further changes require explicit request.**

---

## 14.1 Google Staff Engineer (distributed systems / SRE lens)

**Findings:**
- *Consensus theater.* The original ask implied Raft/consensus among agents. No state in
  the system actually needs multi-writer agreement — a single-writer-per-signal +
  TTL'd-lock design is strictly simpler and more debuggable at this scale.
- *No degradation story.* A pure MARL controller with no fallback is a demo, not a system.
  Any sensor glitch or model bug produces undefined signal behavior — disqualifying for a
  "production-grade" claim.
- *Latency budget was unstated.* "Real-time" was asserted, not measured. Without a
  per-hop budget there is nothing to defend under questioning.
- *K8s-for-two-weeks trap.* A full cloud cluster for a solo 12-week project burns 1–2 weeks
  on cluster ops with no system-design payoff beyond "I used K8s."

**Changes made:** added the safety shield + RL→max-pressure→fixed-time ladder as
non-negotiable (§3.10, §4.1); explicit consensus-avoidance rationale (§5.5); published
latency budget table (§5.7) enforced by tests; scoped K8s to one Helm chart on kind with a
scripted demo, cloud cluster explicitly cut (§10.3, Part 13 Week 10).

## 14.2 OpenAI / Anthropic-style Agentic-AI Engineer

**Findings:**
- *Two orchestration frameworks (LangGraph + CrewAI) is resume keyword-stuffing,* not
  architecture — they'd fight over the same responsibility.
- *Ungrounded LLM agents are a liability, not a feature.* An LLM "reasoning" about traffic
  with no tool access to ground-truth state produces plausible-sounding nonsense under
  pressure — exactly the failure mode a strong interviewer will probe for.
- *Standalone "Weather Agent" / "Construction Agent" processes* that just poll an API and
  forward a message are not agents — they're inflating the agent count for the pitch.
- *$0 LLM budget was not designed around;* the plan silently assumed paid API access.

**Changes made:** cut CrewAI, LangGraph only (§7.3); every Supervisor proposal must cite a
what-if run id, enforced by the graph's structure, not a prompt instruction (§4.4); Weather/
Construction/VIP/Police collapsed into scenario events + playbooks (§4.0); explicit
free-tier provider chain (Groq→Gemini→Ollama→deterministic mapper) with measured quota
headroom, so the system is honestly $0-capable (§4.4, §6.6, §7.4).

## 14.3 NVIDIA (Edge AI / perception / real-time systems lens)

**Findings:**
- *Computer vision was decorative* — a YOLO model bolted on with no connection to the
  control problem reads as checkbox-filling, and reviewers from this world will spot it
  immediately.
- *No edge story* despite "Edge AI" being on the requirements list — no latency/footprint
  numbers, no quantization.
- *No sensing-noise robustness analysis* — the entire MARL literature this project competes
  with already ignores this; repeating the omission wastes the strongest available
  differentiator.

**Changes made:** perception's measured error distribution now parameterizes the twin's
sensor fault model, making CV load-bearing for a first-class robustness experiment (§4.9,
§6.3, §11.5); ONNX INT8 edge profile with published mAP/latency/size table required
deliverable, Week 8 (§6.3, Part 13).

## 14.4 Waymo / Autonomy (safety-critical systems lens)

**Findings:**
- *Learned emergency preemption is a safety anti-pattern.* Preemption is a solved,
  well-understood algorithm; asking an RL policy to learn it (as the original 12-agent list
  implied via "Emergency Vehicle Agent") risks unsafe generalization for a problem with a
  known-good deterministic answer — a real safety engineer would reject this immediately.
- *No measurement of recovery behavior.* Emergency response time alone hides whether the
  network degrades badly afterward.
- *No explicit action-masking / shield co-design with training,* risking a shield that
  fights the policy instead of one the policy is trained under.

**Changes made:** preemption is a deterministic overlay, explicitly not learned, with
rationale documented (§4.3); post-preemption network recovery time added as a first-class
metric (§1.6, §11.2); the shield is unit-tested as a standalone library and wired into
*both* training (masked action distributions) and execution identically, so there's no
train/serve skew (§4.1, Part 13 Week 3).

## 14.5 Microsoft Research (MLOps / evaluation rigor lens)

**Findings:**
- *No evaluation statistics.* "RL beats fixed-time" with no seeds, no confidence intervals,
  no baseline beyond the weakest one, is not a result — it's a screenshot.
- *No model lifecycle.* Training scripts with no registry, no promotion gate, no rollback
  path is not "MLOps," it's a notebook with extra steps.
- *No lineage/reproducibility guarantee* despite datasets, models, and code all changing
  independently over 12 weeks — a recipe for irreproducible headline numbers by Week 12.

**Changes made:** 10-seed, 95%-CI, Holm–Bonferroni-corrected evaluation protocol against
three baselines mandated (Part 11); MLflow registry with Staging→shadow-eval→Production
promotion and CI regression gate (`eval.yml`) added (§10.4, §10.5); every run required to
log `git_sha`+`dvc_hash`+`config_hash`+`seed`, enforced in CI (§8.7).

## 14.6 Amazon (operational excellence / cost lens)

**Findings:**
- *100+ intersection ambition was incompatible with the stated Colab/Kaggle-only compute* —
  a plan that can't run its own training within the available compute is not a plan.
- *No stated compute/cost ceiling* for the LLM layer despite a $0 budget answer — free-tier
  limits (RPM/RPD) needed to be measured against actual expected call volume, not assumed
  sufficient.
- *No chaos/failure testing* — "fault tolerant" was asserted with no test exercising a
  killed process or partitioned bus.

**Changes made:** scale reset to ~25-intersection district (matches the user's own answer)
with an explicit, honest scaling narrative to 500 for the "how would this scale" interview
question instead of pretending to build it (§3.10, §10.8); measured free-tier headroom
against expected demo-day call volume (14.4K/day quota vs. <200 calls/day expected) (§4.4,
§6.6); `tests/chaos/` made a required deliverable with pass criteria, exercised in both
compose and kind CI (§10.4, §11.5, Part 13 Weeks 6–7, 10).

---

## 14.7 Resume / interview-readiness pass

**Weaknesses found in the original scope:**
- A 12-agent, RLlib+CrewAI+full-cloud-K8s system oversells scope a solo 12-week build
  cannot finish — an interviewer who asks "walk me through the hardest bug" would find
  nothing finished enough to walk through.
- No single number in the original plan was defensible under "how do you know that's true?"
- "Agentic AI," "GNN," "Edge AI," "distributed systems," "MLOps" were requirements-list
  keywords with no forcing function connecting them to the core problem — exactly what a
  discerning reviewer discounts.

**Resolution:** the frozen architecture makes every requirement-list keyword *load-bearing*
(GNN feeds real anticipatory decisions; Edge AI has a measured tradeoff table; distributed
systems claims are chaos-tested; MLOps has an enforced regression gate) and matches the
stated 12-week/solo/Colab/$0 constraints exactly, so the finished project is both smaller in
surface area and stronger in defensibility than the original ask. Part 13's roadmap is the
executable proof this fits the budget.

## 14.8 What was explicitly cut and why (final list)

RLlib (own MAPPO instead — depth over configuration); CrewAI (LangGraph only); hand-rolled
Raft/consensus (not needed at this scale, documented where it would enter); 100+
intersection scale (documented scaling path instead); always-on cloud K8s cluster (kind +
portable Helm chart instead); standalone Weather/Construction/VIP/Police agent processes
(scenario events + playbooks); METR-LA as a core deliverable (optional stretch validation
only); learned emergency preemption (deterministic overlay, safety-critical reasoning).

**Status: architecture frozen.** Any future change to the above requires an explicit user
request that supersedes this record.
