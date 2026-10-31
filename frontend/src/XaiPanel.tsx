import type { Incident, TrafficImpact } from "./api";

interface XaiPanelProps {
  incident: Incident | null;
  /** Measured effect of the applied playbook in the live twin, if any. */
  impact?: TrafficImpact | null;
}

const METRIC_LABEL: Record<string, string> = {
  mean_halted_vehicles: "mean halted vehicles",
  mean_pressure: "mean pressure",
};

function percentChange(before: number, after: number): string {
  if (before === 0) return "";
  const pct = ((after - before) / before) * 100;
  return ` (${pct > 0 ? "+" : "−"}${Math.abs(pct).toFixed(0)}%)`;
}

function ImpactBlock({ impact }: { impact: TrafficImpact }) {
  if (impact.status === "not_applied") {
    return <p className="impact">Not applied to the live twin: {impact.summary}.</p>;
  }
  if (impact.status === "no_change") {
    return <p className="impact">Applied: {impact.summary} (no road effect).</p>;
  }
  const before = impact.before_halted;
  const after = impact.after_halted;
  return (
    <div className="impact">
      <h4>Observed impact in the live twin</h4>
      <p>{impact.summary}</p>
      {impact.status === "measuring" ? (
        <p>
          Measuring… {Math.round(impact.elapsed_s)} / {Math.round(impact.window_s)} s of sim time
        </p>
      ) : before !== null && after !== null ? (
        <p>
          Queued vehicles on the affected roads: <b>{before.toFixed(1)}</b> before →{" "}
          <b>{after.toFixed(1)}</b> after{percentChange(before, after)}
        </p>
      ) : (
        <p>Not enough traffic on the affected roads to measure.</p>
      )}
      <p className="empty">One live run, not a controlled test.</p>
    </div>
  );
}

// Every field rendered here is a real value returned by the Cognitive API from a real
// Supervisor graph run (docs/design/04_agent_architecture.md section 4.5's XAI panel)
// -- no narrated/LLM-generated text, since the Analyst/XAI agent's v1 scope
// (src/atlas/cognitive/analyst.py) is deliberately the tool-computed substance,
// not prose. There is no dashboard-facing endpoint for per-decision counterfactual
// probes yet (that needs a live ActorCritic checkpoint wired into the running
// service, not built for Week 9/10) -- documented gap, not faked here.
export function XaiPanel({ incident, impact = null }: XaiPanelProps) {
  if (!incident) {
    return <p className="empty">Select an incident to see why the system responded as it did.</p>;
  }

  const { state } = incident;
  const proposal = state.proposal;
  const detail = state.what_if_detail;

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

      {detail && (
        <p className="what-if-evidence">
          Simulated (same seed, with vs. without the playbook):{" "}
          {METRIC_LABEL[detail.metric] ?? detail.metric} <b>{detail.baseline.toFixed(1)}</b> →{" "}
          <b>{detail.proposed.toFixed(1)}</b>
          {percentChange(detail.baseline, detail.proposed)}
        </p>
      )}

      {proposal && (
        <div className="proposal">
          <h4>Proposal: {proposal.playbook_name}</h4>
          <p className="rationale">{proposal.rationale}</p>
          <pre>{JSON.stringify(proposal.params, null, 2)}</pre>
        </div>
      )}

      {impact && <ImpactBlock impact={impact} />}

      {incident.interrupt && (
        <p className="interrupt-question">{incident.interrupt.question}</p>
      )}
    </div>
  );
}
