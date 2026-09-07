import { useCallback, useEffect, useState } from "react";
import "./App.css";
import { MapView } from "./MapView";
import { ApprovalQueue } from "./ApprovalQueue";
import { XaiPanel } from "./XaiPanel";
import {
  ApiError,
  approveIncident,
  listIncidents,
  rejectIncident,
  submitIncident,
  type Incident,
} from "./api";

const DEFAULT_API_BASE_URL = "http://localhost:8000";
const POLL_INTERVAL_MS = 3000;

// Real-time updates would normally arrive over the `dashboard_api` websocket bridge
// (docs/design/09_repository_design.md's `services/dashboard_api (ws bridge)`) tailing
// the event bus. No live Kafka/Redpanda locally (same documented gap as every other
// bus-based service in this project) -- polling the real Cognitive API on an interval
// is this dashboard's own honest substitute, not a fake live feed.
let demoIncidentCounter = 0;

function App() {
  const [apiBaseUrl, setApiBaseUrl] = useState(DEFAULT_API_BASE_URL);
  const [operatorToken, setOperatorToken] = useState("");
  const [incidents, setIncidents] = useState<Incident[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const result = await listIncidents(apiBaseUrl);
      setIncidents(result);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }, [apiBaseUrl]);

  useEffect(() => {
    refresh();
    const timer = setInterval(refresh, POLL_INTERVAL_MS);
    return () => clearInterval(timer);
  }, [refresh]);

  const selected = incidents.find((i) => i.incident_id === selectedId) ?? null;

  async function handleApprove(incidentId: string) {
    setBusyId(incidentId);
    try {
      await approveIncident(apiBaseUrl, operatorToken, incidentId);
      await refresh();
    } catch (err) {
      setError(err instanceof ApiError ? `Approve failed (${err.status}): ${err.message}` : String(err));
    } finally {
      setBusyId(null);
    }
  }

  async function handleReject(incidentId: string) {
    setBusyId(incidentId);
    try {
      await rejectIncident(apiBaseUrl, operatorToken, incidentId);
      await refresh();
    } catch (err) {
      setError(err instanceof ApiError ? `Reject failed (${err.status}): ${err.message}` : String(err));
    } finally {
      setBusyId(null);
    }
  }

  async function handleSubmitDemoIncident() {
    demoIncidentCounter += 1;
    const incidentId = `demo-incident-${Date.now()}-${demoIncidentCounter}`;
    try {
      await submitIncident(apiBaseUrl, operatorToken, {
        incident_id: incidentId,
        class: "sensor_fault",
        status: "detected",
        confidence: 0.9,
        // Real manhattan_midtown corridor junction (matches MapView.tsx's
        // corridor_geometry.json), not a placeholder id -- the demo API server
        // (scripts/run_dashboard_demo_api.py) runs against this same real network.
        location: { link_ids: ["42430333"] },
      });
      await refresh();
      setSelectedId(incidentId);
    } catch (err) {
      setError(err instanceof ApiError ? `Submit failed (${err.status}): ${err.message}` : String(err));
    }
  }

  return (
    <div className="app">
      <header className="topbar">
        <h1>ATLAS Operations Dashboard</h1>
        <div className="settings">
          <label>
            API base URL
            <input value={apiBaseUrl} onChange={(e) => setApiBaseUrl(e.target.value)} />
          </label>
          <label>
            Operator token
            <input
              type="password"
              placeholder="paste a JWT to approve/reject/submit"
              value={operatorToken}
              onChange={(e) => setOperatorToken(e.target.value)}
            />
          </label>
          <button type="button" onClick={handleSubmitDemoIncident}>
            Submit demo incident
          </button>
        </div>
      </header>

      {error && <div className="error-banner">{error}</div>}

      <main className="layout">
        <MapView incidents={incidents} selectedId={selectedId} onSelectIncident={setSelectedId} />
        <div className="sidebar">
          <section>
            <h2>Approval queue</h2>
            <ApprovalQueue
              incidents={incidents}
              selectedId={selectedId}
              onSelectIncident={setSelectedId}
              onApprove={handleApprove}
              onReject={handleReject}
              busyId={busyId}
            />
          </section>
          <section>
            <h2>Explanation (XAI panel)</h2>
            <XaiPanel incident={selected} />
          </section>
          <section>
            <h2>Replay scrubber</h2>
            <p className="empty">
              Deferred: needs the Timescale-backed telemetry store
              (docs/design/03_system_architecture.md section 3.5), not built yet.
            </p>
          </section>
        </div>
      </main>
    </div>
  );
}

export default App;
