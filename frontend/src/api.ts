// Thin fetch wrapper around the real Cognitive API (src/atlas/services/cognitive_api.py).
// No mock data anywhere here -- every function below hits a real backend endpoint;
// if the API isn't running, calls fail visibly (see App.tsx's error state), not
// silently substituted with fixtures.

export interface Proposal {
  playbook_name: string;
  params: Record<string, unknown>;
  rationale: string;
}

export interface IncidentState {
  incident: Record<string, unknown>;
  proposal: Proposal | null;
  provider_used: string | null;
  what_if_checked: boolean;
  what_if_improved: boolean | null;
  auto_applied: boolean;
  approved: boolean | null;
  status: string;
}

export interface Incident {
  incident_id: string;
  state: IncidentState;
  pending_approval: boolean;
  interrupt: { question: string } | null;
}

// Live traffic from the real digital twin (src/atlas/services/live_traffic.py).
export interface TrafficEdgeGeometry {
  edge_id: string;
  lanes: number;
  speed_limit_mps: number;
  /** [lon, lat] points along the road. */
  coords: [number, number][];
}

export interface TrafficEdgeStat {
  vehicles: number;
  /** Mean speed / speed limit, 0..1; null when no vehicle is on the road. */
  speed_ratio: number | null;
}

export interface TrafficDisruption {
  name: string;
  type: string;
  edge_id: string;
  incident_id: string;
  ends_at_sim_s: number;
}

export interface TrafficSnapshot {
  ready: boolean;
  error: string | null;
  sim_time_s: number;
  stats: { vehicles: number; halted: number; mean_speed_kmh: number };
  edges: Record<string, TrafficEdgeStat>;
  /** [lon, lat, speed_mps] per vehicle. */
  vehicles: [number, number, number][];
  disruptions: TrafficDisruption[];
}

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

async function request<T>(
  baseUrl: string,
  path: string,
  init?: RequestInit,
): Promise<T> {
  const response = await fetch(`${baseUrl}${path}`, init);
  if (!response.ok) {
    const detail = await response.text();
    throw new ApiError(detail || response.statusText, response.status);
  }
  return (await response.json()) as T;
}

export function listIncidents(baseUrl: string): Promise<Incident[]> {
  return request<Incident[]>(baseUrl, "/incidents");
}

export function getTrafficGeometry(baseUrl: string): Promise<{ edges: TrafficEdgeGeometry[] }> {
  return request<{ edges: TrafficEdgeGeometry[] }>(baseUrl, "/traffic/geometry");
}

export function getTraffic(baseUrl: string): Promise<TrafficSnapshot> {
  return request<TrafficSnapshot>(baseUrl, "/traffic");
}

function authHeaders(token: string): HeadersInit {
  return { Authorization: `Bearer ${token}`, "Content-Type": "application/json" };
}

export function submitIncident(
  baseUrl: string,
  token: string,
  incident: Record<string, unknown>,
): Promise<Incident> {
  return request<Incident>(baseUrl, "/incidents", {
    method: "POST",
    headers: authHeaders(token),
    body: JSON.stringify(incident),
  });
}

export function approveIncident(
  baseUrl: string,
  token: string,
  incidentId: string,
): Promise<Incident> {
  return request<Incident>(baseUrl, `/incidents/${incidentId}/approve`, {
    method: "POST",
    headers: authHeaders(token),
  });
}

export function rejectIncident(
  baseUrl: string,
  token: string,
  incidentId: string,
): Promise<Incident> {
  return request<Incident>(baseUrl, `/incidents/${incidentId}/reject`, {
    method: "POST",
    headers: authHeaders(token),
  });
}
