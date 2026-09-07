import type { Incident } from "./api";

interface XaiPanelProps {
  incident: Incident | null;
}

// Every field rendered here is a real value returned by the Cognitive API from a real
// Supervisor graph run (docs/design/04_agent_architecture.md section 4.5's XAI panel)
// -- no narrated/LLM-generated text, since the Analyst/XAI agent's v1 scope
// (src/atlas/cognitive/analyst.py) is deliberately the tool-computed substance,
// not prose. There is no dashboard-facing endpoint for per-decision counterfactual
// probes yet (that needs a live ActorCritic checkpoint wired into the running
// service, not built for Week 9/10) -- documented gap, not faked here.
export function XaiPanel({ incident }: XaiPanelProps) {
  if (!incident) {
    return <p className="empty">Select an incident to see why the system responded as it did.</p>;
  }

  const { state } = incident;
  const proposal = state.proposal;

  return (
    <div className="xai-panel">
      <h3>{incident.incident_id}</h3>
      <dl>
        <dt>Status</dt>
        <dd>{state.status}</dd>
        <dt>Provider</dt>
        <dd>{state.provider_used ?? "—"}</dd>
        <dt>What-if checked</dt>
        <dd>{String(state.what_if_checked)}</dd>
        <dt>What-if improved</dt>
        <dd>{state.what_if_improved === null ? "—" : String(state.what_if_improved)}</dd>
        <dt>Auto-applied</dt>
        <dd>{String(state.auto_applied)}</dd>
        <dt>Approved</dt>
        <dd>{state.approved === null ? "pending" : String(state.approved)}</dd>
      </dl>

      {proposal && (
        <div className="proposal">
          <h4>Proposal: {proposal.playbook_name}</h4>
          <p className="rationale">{proposal.rationale}</p>
          <pre>{JSON.stringify(proposal.params, null, 2)}</pre>
        </div>
      )}

      {incident.interrupt && (
        <p className="interrupt-question">{incident.interrupt.question}</p>
      )}
    </div>
  );
}
