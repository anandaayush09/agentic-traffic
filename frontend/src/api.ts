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
  /** The simulated comparison behind `what_if_improved`; absent in older snapshots. */
  what_if_detail?: { metric: string; baseline: number; proposed: number } | null;
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

export interface TrafficJunction {
  /** The controller currently driving the junction. */
  mode: string;
  /** True while an applied playbook is overriding the operator-chosen mode. */
  playbook_override: boolean;
  phase: number;
  /** Halted vehicles on the junction's approaches. */
  queue: number;
}

export interface TrafficPlaybook {
  incident_id: string;
  playbook: string;
  summary: string;
  ends_at_sim_s: number;
}

/** Observed before/after queues around one applied playbook (one live run, not a
 * controlled experiment -- the what-if tool is the controlled comparison). */
export interface TrafficImpact {
  incident_id: string;
  playbook: string;
  summary: string;
  status: "measuring" | "complete" | "no_change" | "not_applied";
  edge_ids: string[];
  window_s: number;
  elapsed_s: number;
  before_halted: number | null;
  after_halted: number | null;
  before_speed_ratio: number | null;
  after_speed_ratio: number | null;
  halted_change_pct: number | null;
}

export interface JunctionPosition {
  id: string;
  lon: number;
  lat: number;
}

export interface TrafficSnapshot {
  ready: boolean;
  error: string | null;
  sim_time_s: number;
  stats: {
    vehicles: number;
    halted: number;
    mean_speed_kmh: number;
    /** Simulated seconds per wall second; below 1 the map runs slower than reality. */
    x_real_time?: number | null;
    /** Vehicles actually drawn (a city caps this); `vehicles` is the exact total. */
    shown_vehicles?: number;
  };
  edges: Record<string, TrafficEdgeStat>;
  /** [lon, lat, speed_mps] per vehicle. */
  vehicles: [number, number, number][];
  disruptions: TrafficDisruption[];
  junctions: Record<string, TrafficJunction>;
  /** Controller modes the API accepts right now (`rl` only when a policy is loaded). */
  available_modes?: string[];
  playbooks: TrafficPlaybook[];
  impacts: Record<string, TrafficImpact>;
}

/** The classical modes that always exist; the live list comes from the snapshot. */
export const DEFAULT_CONTROLLER_MODES = ["fixed_time", "actuated", "max_pressure"];
export type ControllerMode = string;

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

export function getTrafficGeometry(
  baseUrl: string,
): Promise<{ edges: TrafficEdgeGeometry[]; junctions?: JunctionPosition[] }> {
  return request<{ edges: TrafficEdgeGeometry[]; junctions?: JunctionPosition[] }>(
    baseUrl,
    "/traffic/geometry",
  );
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

export function getTrafficHistory(
  baseUrl: string,
  frames = 300,
  stride = 2,
): Promise<{ frames: TrafficSnapshot[] }> {
  return request<{ frames: TrafficSnapshot[] }>(
    baseUrl,
    `/traffic/history?frames=${frames}&stride=${stride}`,
  );
}

export function getAuthConfig(baseUrl: string): Promise<{ demo_token_enabled: boolean }> {
  return request<{ demo_token_enabled: boolean }>(baseUrl, "/auth/config");
}

/** Local-dev convenience: only works when the API was started with the demo token enabled. */
export async function requestDemoToken(baseUrl: string): Promise<string> {
  const result = await request<{ token: string }>(baseUrl, "/auth/demo-token", { method: "POST" });
  return result.token;
}

export function trafficSocketUrl(baseUrl: string): string {
  return `${baseUrl.replace(/^http/, "ws")}/ws/traffic`;
}

export function setJunctionMode(
  baseUrl: string,
  token: string,
  tlsId: string,
  mode: ControllerMode,
): Promise<{ tls_id: string; mode: string }> {
  return request<{ tls_id: string; mode: string }>(
    baseUrl,
    `/traffic/junctions/${encodeURIComponent(tlsId)}/mode`,
    { method: "POST", headers: authHeaders(token), body: JSON.stringify({ mode }) },
  );
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
