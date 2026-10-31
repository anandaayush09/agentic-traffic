import { useEffect, useState } from "react";
import {
  getTraffic,
  getTrafficGeometry,
  trafficSocketUrl,
  type JunctionPosition,
  type TrafficEdgeGeometry,
  type TrafficSnapshot,
} from "./api";

const POLL_INTERVAL_MS = 1000;
const RECONNECT_DELAY_MS = 5000;

export type TrafficTransport = "websocket" | "polling";

interface TrafficFeed {
  roads: TrafficEdgeGeometry[];
  /** Where every ATLAS-controlled junction is; empty until the geometry has loaded. */
  junctions: JunctionPosition[];
  traffic: TrafficSnapshot | null;
  /** True when the API has no live twin (or can't be reached). Never an error banner:
   * live traffic is optional, the incident workflow works without it. */
  unavailable: boolean;
  transport: TrafficTransport;
}

/**
 * Live traffic for the map. Prefers a WebSocket (the server pushes each new snapshot, no
 * request per second); if the socket can't open or drops, it falls back to polling
 * `GET /traffic` every second and keeps retrying the socket, so a flaky connection
 * degrades the feed instead of freezing it. Road geometry is static and fetched once.
 */
export function useTrafficFeed(apiBaseUrl: string, enabled: boolean): TrafficFeed {
  const [roads, setRoads] = useState<TrafficEdgeGeometry[]>([]);
  const [junctions, setJunctions] = useState<JunctionPosition[]>([]);
  const [traffic, setTraffic] = useState<TrafficSnapshot | null>(null);
  const [unavailable, setUnavailable] = useState(false);
  const [transport, setTransport] = useState<TrafficTransport>("polling");

  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    let haveGeometry = false;
    let socketOpen = false;
    let socket: WebSocket | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout> | undefined;

    async function ensureGeometry() {
      if (haveGeometry) return;
      const geometry = await getTrafficGeometry(apiBaseUrl);
      if (cancelled) return;
      setRoads(geometry.edges);
      setJunctions(geometry.junctions ?? []);
      haveGeometry = true;
    }

    async function poll() {
      try {
        await ensureGeometry();
        if (socketOpen) return; // the socket is delivering; don't double up
        const snapshot = await getTraffic(apiBaseUrl);
        if (cancelled) return;
        setTraffic(snapshot);
        setUnavailable(false);
      } catch {
        if (!cancelled) setUnavailable(true);
      }
    }

    function connect() {
      if (cancelled) return;
      try {
        socket = new WebSocket(trafficSocketUrl(apiBaseUrl));
      } catch {
        reconnectTimer = setTimeout(connect, RECONNECT_DELAY_MS);
        return;
      }
      socket.onopen = () => {
        socketOpen = true;
        setTransport("websocket");
      };
      socket.onmessage = (event: MessageEvent<string>) => {
        if (cancelled) return;
        setTraffic(JSON.parse(event.data) as TrafficSnapshot);
        setUnavailable(false);
      };
      socket.onclose = () => {
        socketOpen = false;
        if (cancelled) return;
        setTransport("polling");
        reconnectTimer = setTimeout(connect, RECONNECT_DELAY_MS);
      };
      socket.onerror = () => socket?.close();
    }

    poll();
    const pollTimer = setInterval(poll, POLL_INTERVAL_MS);
    connect();

    return () => {
      cancelled = true;
      clearInterval(pollTimer);
      clearTimeout(reconnectTimer);
      if (socket) {
        socket.onclose = null;
        socket.close();
      }
    };
  }, [apiBaseUrl, enabled]);

  return { roads, junctions, traffic, unavailable, transport };
}
