import type { TrafficDisruption, TrafficSnapshot } from "./api";

interface TrafficOverlayProps {
  traffic: TrafficSnapshot | null;
  /** True when the API has no live twin attached (or can't be reached). */
  unavailable: boolean;
}

function formatClock(totalSeconds: number): string {
  const s = Math.max(0, Math.round(totalSeconds));
  const minutes = Math.floor(s / 60);
  const seconds = s % 60;
  return `${minutes}:${String(seconds).padStart(2, "0")}`;
}

// One line per incident (an incident can disrupt several roads at once).
function groupByIncident(disruptions: TrafficDisruption[]) {
  const groups = new Map<string, { type: string; roads: number; endsAt: number }>();
  for (const d of disruptions) {
    const group = groups.get(d.incident_id);
    if (group) {
      group.roads += 1;
    } else {
      groups.set(d.incident_id, { type: d.type, roads: 1, endsAt: d.ends_at_sim_s });
    }
  }
  return [...groups.entries()];
}

export function TrafficOverlay({ traffic, unavailable }: TrafficOverlayProps) {
  if (unavailable) {
    return (
      <div className="traffic-overlay">
        <strong>Live traffic unavailable</strong>
        <p className="traffic-note">
          The API has no live simulation attached (or can't be reached). Incident status
          markers still work.
        </p>
      </div>
    );
  }
  if (!traffic || !traffic.ready) {
    return (
      <div className="traffic-overlay">
        <strong>Live traffic</strong>
        <p className="traffic-note">
          {traffic?.error ? `Simulation error: ${traffic.error}` : "Simulation warming up…"}
        </p>
      </div>
    );
  }

  const incidents = groupByIncident(traffic.disruptions);
  return (
    <div className="traffic-overlay">
      <strong>Live traffic · real SUMO simulation</strong>
      <div className="traffic-stats">
        <span>Sim clock {formatClock(traffic.sim_time_s)}</span>
        <span>{traffic.stats.vehicles} vehicles</span>
        <span>{traffic.stats.halted} stopped</span>
        <span>avg {traffic.stats.mean_speed_kmh} km/h</span>
      </div>
      <ul className="traffic-legend">
        <li>
          <i style={{ background: "#2ecc71" }} /> free flow
        </li>
        <li>
          <i style={{ background: "#f0ad4e" }} /> slow
        </li>
        <li>
          <i style={{ background: "#d9534f" }} /> congested
        </li>
        <li>
          <i style={{ background: "#8d99a6" }} /> empty
        </li>
        <li>
          <i className="legend-disruption" /> disrupted road
        </li>
      </ul>
      {incidents.length > 0 && (
        <ul className="traffic-disruptions">
          {incidents.map(([incidentId, g]) => (
            <li key={incidentId}>
              <b>{g.type}</b> on {g.roads} road{g.roads === 1 ? "" : "s"} ({incidentId}) — clears
              in {formatClock(g.endsAt - traffic.sim_time_s)} sim time
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
