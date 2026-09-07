"""The Analyst / XAI Agent v1 (docs/design/04_agent_architecture.md section 4.5): "Turn
decision logs into operator-readable explanations... numeric claims are computed by
tools..., not by the model" and section 6.7's "(1) decision logs... (2) counterfactual
probes... (3) LLM narration strictly over (1)+(2)".

v1 scope, deliberately: this module IS (1) and (2) -- the tool-computed substance an LLM
would narrate over. No LLM is involved yet, and none is needed for this to be useful:
every explanation here quotes REAL logged values and a REAL re-run of the policy, never
a guessed number. Wiring an LLM on top to phrase these facts more fluently is a
narration layer over already-correct output, not a prerequisite for correctness --
matching how the Supervisor's deterministic mapper (Week 9, part 3) is itself a
complete, correct fallback, not a stub waiting for an LLM.

Refusal (section 4.5: "refuses... when logs don't cover the question -- hallucinated
explanations are worse than none") is load-bearing here, not decorative: `explain`
raises `DecisionNotFoundError` for an unlogged decision_id rather than returning
anything at all, so a caller (an LLM narration layer, or a human) can never mistake
"nothing was found" for "here is an answer".
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from atlas.marl.network import ActorCritic


class DecisionNotFoundError(KeyError):
    """No logged decision matches the given id -- the refusal case section 4.5 asks
    for, made a distinct exception rather than returning None/empty string so a caller
    cannot accidentally treat "not found" as a valid (if boring) explanation."""


@dataclass(frozen=True, slots=True)
class DecisionLogEntry:
    """One control.commands.v1-shaped decision record -- the actual input this module
    explains. Deliberately the same fields that topic already carries (Week 5), so a
    real audit consumer needs no new data it doesn't already have."""

    decision_id: str
    intersection_id: str
    actor: str
    phase_id: int
    shield_modified: bool
    shield_reason: str | None
    pressure: float


class DecisionLogStore:
    """An in-memory decision log -- the swap point for a real Postgres-backed audit
    store (docs/design/04_agent_architecture.md section 4.4's "incident journal in
    Postgres"), same interim-implementation pattern as InMemoryLockStore/
    InMemoryDeduplicator elsewhere in this project.
    """

    def __init__(self) -> None:
        self._entries: dict[str, DecisionLogEntry] = {}

    def record(self, entry: DecisionLogEntry) -> None:
        self._entries[entry.decision_id] = entry

    def get(self, decision_id: str) -> DecisionLogEntry:
        if decision_id not in self._entries:
            raise DecisionNotFoundError(
                f"No logged decision with id '{decision_id}' -- cannot explain what was never recorded."
            )
        return self._entries[decision_id]


def explain_decision(entry: DecisionLogEntry) -> str:
    """A template-guided explanation quoting the entry's own logged values directly --
    no numeric claim here is invented; every one is read straight from `entry`.
    """
    if entry.shield_modified:
        return (
            f"At junction '{entry.intersection_id}', the '{entry.actor}' controller's "
            f"requested phase was OVERRIDDEN by the safety shield (reason: "
            f"'{entry.shield_reason}'). The junction was set to phase {entry.phase_id}. "
            f"Logged pressure at this decision: {entry.pressure:.2f}."
        )
    return (
        f"At junction '{entry.intersection_id}', the '{entry.actor}' controller chose "
        f"phase {entry.phase_id}, applied as requested (the shield did not need to "
        f"intervene). Logged pressure at this decision: {entry.pressure:.2f}."
    )


@dataclass(frozen=True, slots=True)
class CounterfactualResult:
    original_action: int
    perturbed_action: int
    action_changed: bool
    description: str


def run_counterfactual(
    net: ActorCritic, original_obs: torch.Tensor, feature_index: int, new_value: float
) -> CounterfactualResult:
    """ "With the queue on the north approach halved, the policy chooses phase 2"
    (section 4.5's own example) made literal: re-runs `net`'s actor forward pass with
    ONE feature of `original_obs` replaced by `new_value`, and reports whether the
    chosen action actually changes. A real re-evaluation of the real policy, not a
    guessed sensitivity.
    """
    if not 0 <= feature_index < original_obs.shape[-1]:
        raise ValueError(
            f"feature_index {feature_index} out of range for obs of shape {original_obs.shape}"
        )

    perturbed_obs = original_obs.clone()
    perturbed_obs[feature_index] = new_value

    with torch.no_grad():
        original_action, _, _ = net.act(original_obs.unsqueeze(0), deterministic=True)
        perturbed_action, _, _ = net.act(perturbed_obs.unsqueeze(0), deterministic=True)

    original = int(original_action.item())
    perturbed = int(perturbed_action.item())
    changed = original != perturbed
    description = (
        f"Setting observation feature {feature_index} to {new_value} "
        f"{'CHANGES' if changed else 'does not change'} the policy's chosen action "
        f"(originally phase {original}, {'now phase ' + str(perturbed) if changed else 'stays phase ' + str(original)})."
    )
    return CounterfactualResult(
        original_action=original,
        perturbed_action=perturbed,
        action_changed=changed,
        description=description,
    )


class AnalystAgent:
    """Ties the decision log and counterfactual tool together behind one `ask`
    interface -- the "template-guided generation with tool calls" section 4.5
    describes, minus the LLM (v1 scope, see module docstring).
    """

    def __init__(self, log_store: DecisionLogStore) -> None:
        self._log_store = log_store

    def explain(self, decision_id: str) -> str:
        entry = self._log_store.get(decision_id)  # raises DecisionNotFoundError if absent
        return explain_decision(entry)

    def what_if_feature_changed(
        self,
        net: ActorCritic,
        original_obs: torch.Tensor,
        feature_index: int,
        new_value: float,
    ) -> CounterfactualResult:
        return run_counterfactual(net, original_obs, feature_index, new_value)
