import { useCallback, useEffect, useState } from "react";
import "./App.css";
import { MapView } from "./MapView";
import { ApprovalQueue } from "./ApprovalQueue";
import { XaiPanel } from "./XaiPanel";
import staticSnapshot from "./public_demo_snapshot.json";
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

// The public read-only demo (docs/design/10_deployment.md: "zero control endpoints,
// no bus, no LLM keys") is a SEPARATE build of this same app, not a config flag on
// the live one: `npm run build:demo` sets VITE_STATIC_DEMO=true, which makes this
// component load a real, frozen snapshot (scripts/export_public_demo_snapshot.py --
// real Supervisor graph runs against the real manhattan_midtown corridor, not
// fabricated data) with zero network calls and no mutating UI at all. The live/dev
// build (`npm run dev` / `npm run build`) is unaffected -- polls a real Cognitive
// API and keeps full approve/reject wiring, exactly as before.
const STATIC_DEMO = import.meta.env.VITE_STATIC_DEMO === "true";
const STATIC_INCIDENTS = staticSnapshot as Incident[];

// Real-time updates would normally arrive over the `dashboard_api` websocket bridge
// (docs/design/09_repository_design.md's `services/dashboard_api (ws bridge)`) tailing
// the event bus. No live Kafka/Redpanda locally (same documented gap as every other
// bus-based service in this project) -- polling the real Cognitive API on an interval
// is this dashboard's own honest substitute, not a fake live feed.
let demoIncidentCounter = 0;

function App() {
  const [apiBaseUrl, setApiBaseUrl] = useState(DEFAULT_API_BASE_URL);
  const [operatorToken, setOperatorToken] = useState("");
  const [incidents, setIncidents] = useState<Incident[]>(STATIC_DEMO ? STATIC_INCIDENTS : []);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    if (STATIC_DEMO) return; // nothing to poll -- the snapshot is loaded once, above
    try {
      const result = await listIncidents(apiBaseUrl);
      setIncidents(result);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }, [apiBaseUrl]);

  useEffect(() => {
    if (STATIC_DEMO) return;
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
        {STATIC_DEMO ? (
          <div className="settings">
            <span className="demo-banner">
              Public read-only demo — replaying {STATIC_INCIDENTS.length} real recorded
              incidents (no live backend, no control actions)
            </span>
          </div>
        ) : (
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
        )}
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
              readOnly={STATIC_DEMO}
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
