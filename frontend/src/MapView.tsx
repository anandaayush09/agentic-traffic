import { useEffect, useRef } from "react";
import {
  LngLatBounds,
  Map as MapLibreMap,
  Marker,
  type ExpressionSpecification,
  type GeoJSONSource,
} from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import corridorGeometry from "./corridor_geometry.json";
import type { Incident, JunctionPosition, TrafficEdgeGeometry, TrafficSnapshot } from "./api";
import { TrafficOverlay } from "./TrafficOverlay";

// Real junction coordinates for the Week 5 corridor (manhattan_midtown, 5 contiguous
// signalized junctions), exported from the actual built SUMO network by
// scripts/export_corridor_geometry.py -- not hand-placed pins.
const JUNCTIONS: { intersection_id: string; lon: number; lat: number }[] =
  corridorGeometry.junctions;

const ROADS_SOURCE = "roads";
const VEHICLES_SOURCE = "vehicles";
const JUNCTIONS_SOURCE = "junction-points";

// Above this many junctions DOM markers (one element each, repositioned on every pan/zoom) stop
// being viable; a whole city is drawn as one GPU circle layer instead.
const LAYER_JUNCTION_THRESHOLD = 150;

// Per-road values live in MapLibre feature-state, not in the GeoJSON: the (large) road geometry
// is uploaded once, and each tick only the roads whose value changed are touched.
const ROAD_RATIO: ExpressionSpecification = ["coalesce", ["feature-state", "ratio"], -1];

// Road colour = how close traffic is to the speed limit (mean speed / limit):
// grey = empty road, red < 35%, amber 35-65%, green >= 65%.
const ROAD_COLOR: ExpressionSpecification = [
  "case",
  ["<", ROAD_RATIO, 0],
  "#8d99a6",
  ["step", ROAD_RATIO, "#d9534f", 0.35, "#f0ad4e", 0.65, "#2ecc71"],
];

const ROAD_WIDTH: ExpressionSpecification = [
  "interpolate",
  ["linear"],
  ["zoom"],
  13,
  2,
  16,
  5,
  18,
  9,
];

interface FeatureCollectionLike {
  type: "FeatureCollection";
  features: unknown[];
}

type GeoJsonData = Parameters<GeoJSONSource["setData"]>[0];

const EMPTY_COLLECTION: FeatureCollectionLike = { type: "FeatureCollection", features: [] };

function buildRoadGeometry(roads: TrafficEdgeGeometry[]): FeatureCollectionLike {
  return {
    type: "FeatureCollection",
    features: roads.map((road) => ({
      type: "Feature",
      properties: { edge_id: road.edge_id },
      geometry: { type: "LineString", coordinates: road.coords },
    })),
  };
}

interface RoadState {
  ratio: number;
  disrupted: number;
}

function desiredRoadStates(traffic: TrafficSnapshot | null): Map<string, RoadState> {
  const states = new Map<string, RoadState>();
  for (const [edgeId, stat] of Object.entries(traffic?.edges ?? {})) {
    if (stat.speed_ratio != null) states.set(edgeId, { ratio: stat.speed_ratio, disrupted: 0 });
  }
  for (const d of traffic?.disruptions ?? []) {
    states.set(d.edge_id, { ratio: states.get(d.edge_id)?.ratio ?? -1, disrupted: 1 });
  }
  return states;
}

function buildJunctionFeatures(
  points: { intersection_id: string; lon: number; lat: number }[],
  affected: Map<string, Incident>,
  selectedId: string | null,
): FeatureCollectionLike {
  return {
    type: "FeatureCollection",
    features: points.map((point) => {
      const incident = affected.get(point.intersection_id);
      return {
        type: "Feature",
        properties: {
          id: point.intersection_id,
          color: incident ? (incident.pending_approval ? "#d9534f" : "#5cb85c") : "#337ab7",
          big: incident?.incident_id === selectedId ? 1 : 0,
          incident: incident ? 1 : 0,
        },
        geometry: { type: "Point", coordinates: [point.lon, point.lat] },
      };
    }),
  };
}

function buildVehicleFeatures(traffic: TrafficSnapshot | null): FeatureCollectionLike {
  return {
    type: "FeatureCollection",
    features: (traffic?.vehicles ?? []).map(([lon, lat, speed]) => ({
      type: "Feature",
      properties: { v: speed },
      geometry: { type: "Point", coordinates: [lon, lat] },
    })),
  };
}

function affectedJunctionIds(incidents: Incident[]): Map<string, Incident> {
  const affected = new Map<string, Incident>();
  for (const incident of incidents) {
    const location = incident.state.incident["location"] as
      | { link_ids?: unknown }
      | undefined;
    const linkIds = Array.isArray(location?.link_ids) ? location.link_ids : [];
    for (const linkId of linkIds) {
      if (typeof linkId === "string") affected.set(linkId, incident);
    }
  }
  return affected;
}

interface MapViewProps {
  incidents: Incident[];
  selectedId: string | null;
  onSelectIncident: (incidentId: string) => void;
  /** Static road shapes from the live twin; empty when live traffic is unavailable. */
  roads: TrafficEdgeGeometry[];
  /** Every ATLAS-controlled junction from the live twin; falls back to the 5 corridor
   * junctions baked into the build when the API has not supplied them. */
  junctions: JunctionPosition[];
  traffic: TrafficSnapshot | null;
  /** True when the API has no live twin (or can't be reached). */
  trafficUnavailable: boolean;
  /** False in the static public demo build, which has no live backend at all. */
  showTrafficOverlay: boolean;
  /** True while `traffic` is a recorded frame rather than the live snapshot. */
  replay?: boolean;
}

export function MapView({
  incidents,
  selectedId,
  onSelectIncident,
  roads,
  junctions,
  traffic,
  trafficUnavailable,
  showTrafficOverlay,
  replay = false,
}: MapViewProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MapLibreMap | null>(null);
  const markersRef = useRef<Map<string, Marker>>(new Map());
  const styleLoadedRef = useRef(false);
  const trafficDataRef = useRef({ roads, traffic });
  const roadsLoadedRef = useRef<TrafficEdgeGeometry[] | null>(null);
  const roadStatesRef = useRef<Map<string, RoadState>>(new Map());
  const junctionDataRef = useRef({ incidents, selectedId, junctions, onSelectIncident });

  // Pushes the latest roads/vehicles into the map. Safe to call at any time: it does
  // nothing until the basemap style has loaded, and the style's `style.load` handler calls it
  // again so data that arrived first is not lost.
  function syncTraffic() {
    const map = mapRef.current;
    if (!map || !styleLoadedRef.current) return;
    const { roads: currentRoads, traffic: currentTraffic } = trafficDataRef.current;
    if (currentRoads.length === 0) return;

    if (!map.getSource(ROADS_SOURCE)) {
      map.addSource(ROADS_SOURCE, {
        type: "geojson",
        data: EMPTY_COLLECTION as GeoJsonData,
        promoteId: "edge_id",
      });
      map.addSource(VEHICLES_SOURCE, { type: "geojson", data: EMPTY_COLLECTION as GeoJsonData });
      map.addLayer({
        id: "roads-disruption",
        type: "line",
        source: ROADS_SOURCE,
        layout: { "line-cap": "round" },
        paint: {
          "line-color": "#7b2cbf",
          "line-opacity": ["case", ["==", ["feature-state", "disrupted"], 1], 0.55, 0],
          "line-width": 14,
        },
      });
      map.addLayer({
        id: "roads-casing",
        type: "line",
        source: ROADS_SOURCE,
        layout: { "line-cap": "round" },
        paint: { "line-color": "#ffffff", "line-width": ROAD_WIDTH, "line-gap-width": 2 },
      });
      map.addLayer({
        id: "roads",
        type: "line",
        source: ROADS_SOURCE,
        layout: { "line-cap": "round" },
        paint: { "line-color": ROAD_COLOR, "line-width": ROAD_WIDTH },
      });
      map.addLayer({
        id: "vehicles",
        type: "circle",
        source: VEHICLES_SOURCE,
        paint: {
          "circle-radius": ["interpolate", ["linear"], ["zoom"], 13, 1.5, 16, 3, 18, 5],
          "circle-color": ["case", ["<", ["get", "v"], 0.1], "#8b0000", "#1b2a41"],
          "circle-stroke-color": "#ffffff",
          "circle-stroke-width": 0.8,
        },
      });

      const bounds = new LngLatBounds();
      for (const road of currentRoads) {
        for (const coord of road.coords) bounds.extend(coord);
      }
      map.fitBounds(bounds, { padding: 60, duration: 0, maxZoom: 17 });
    }

    const roadsSource = map.getSource(ROADS_SOURCE) as GeoJSONSource;
    if (roadsLoadedRef.current !== currentRoads) {
      roadsSource.setData(buildRoadGeometry(currentRoads) as GeoJsonData);
      roadsLoadedRef.current = currentRoads;
      roadStatesRef.current = new Map();
    }
    const previous = roadStatesRef.current;
    const next = desiredRoadStates(currentTraffic);
    for (const edgeId of previous.keys()) {
      if (!next.has(edgeId)) map.removeFeatureState({ source: ROADS_SOURCE, id: edgeId });
    }
    for (const [edgeId, state] of next) {
      const before = previous.get(edgeId);
      if (!before || before.ratio !== state.ratio || before.disrupted !== state.disrupted) {
        map.setFeatureState({ source: ROADS_SOURCE, id: edgeId }, state);
      }
    }
    roadStatesRef.current = next;
    (map.getSource(VEHICLES_SOURCE) as GeoJSONSource).setData(
      buildVehicleFeatures(currentTraffic) as GeoJsonData,
    );
  }

  // Draws the junction set as one circle layer (city scale). Like syncTraffic it is safe to call
  // at any time and reads refs only.
  function syncJunctionLayer() {
    const map = mapRef.current;
    if (!map || !styleLoadedRef.current) return;
    const { incidents: current, selectedId: selected, junctions: points } = junctionDataRef.current;
    if (points.length <= LAYER_JUNCTION_THRESHOLD) return;
    const features = buildJunctionFeatures(
      points.map((j) => ({ intersection_id: j.id, lon: j.lon, lat: j.lat })),
      affectedJunctionIds(current),
      selected,
    ) as GeoJsonData;
    const existing = map.getSource(JUNCTIONS_SOURCE) as GeoJSONSource | undefined;
    if (existing) {
      existing.setData(features);
      return;
    }
    map.addSource(JUNCTIONS_SOURCE, { type: "geojson", data: features });
    map.addLayer({
      id: "junction-points",
      type: "circle",
      source: JUNCTIONS_SOURCE,
      paint: {
        "circle-radius": [
          "interpolate",
          ["linear"],
          ["zoom"],
          11,
          ["case", ["==", ["get", "incident"], 1], 5, 1.5],
          15,
          ["case", ["==", ["get", "big"], 1], 11, ["==", ["get", "incident"], 1], 8, 4],
          18,
          ["case", ["==", ["get", "big"], 1], 15, 8],
        ],
        "circle-color": ["get", "color"],
        "circle-stroke-color": "#ffffff",
        "circle-stroke-width": 1,
      },
    });
    map.on("click", "junction-points", (event) => {
      const id = event.features?.[0]?.properties?.["id"];
      if (typeof id !== "string") return;
      const { incidents: live, onSelectIncident: select } = junctionDataRef.current;
      const incident = affectedJunctionIds(live).get(id);
      if (incident) select(incident.incident_id);
    });
    map.on("mouseenter", "junction-points", () => (map.getCanvas().style.cursor = "pointer"));
    map.on("mouseleave", "junction-points", () => (map.getCanvas().style.cursor = ""));
  }

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;
    const center: [number, number] = [JUNCTIONS[2].lon, JUNCTIONS[2].lat];
    const map = new MapLibreMap({
      container: containerRef.current,
      // Free, no-API-key OpenStreetMap raster basemap so the street grid is visible
      // at junction zoom (the old maplibre demo-tiles style is country outlines only,
      // blank at street level). Light dev/demo use; keeps the attribution below.
      style: {
        version: 8,
        sources: {
          osm: {
            type: "raster",
            tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
            tileSize: 256,
            attribution: "© OpenStreetMap contributors",
          },
        },
        layers: [{ id: "osm", type: "raster", source: "osm" }],
      },
      center,
      zoom: 15,
    });
    mapRef.current = map;
    // Dev-only handle for inspecting layers from the browser console.
    if (import.meta.env.DEV) (window as unknown as { __atlasMap?: MapLibreMap }).__atlasMap = map;
    // `style.load`, not `load`: `load` also waits for the first basemap tiles, which
    // would hold the traffic layers back (or hide them for good if tiles are blocked).
    map.on("style.load", () => {
      if (mapRef.current !== map) return;
      styleLoadedRef.current = true;
      syncTraffic();
      syncJunctionLayer();
    });
    return () => {
      // Markers belong to this map instance. Drop them with it so a remount (React
      // StrictMode runs effects twice in dev) re-creates them on the new map instead
      // of leaving them attached to the one that was just removed.
      markersRef.current.forEach((marker) => marker.remove());
      markersRef.current.clear();
      styleLoadedRef.current = false;
      roadsLoadedRef.current = null;
      roadStatesRef.current = new Map();
      mapRef.current?.remove();
      mapRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- map is created once; syncTraffic reads refs only
  }, []);

  useEffect(() => {
    trafficDataRef.current = { roads, traffic };
    syncTraffic();
    // eslint-disable-next-line react-hooks/exhaustive-deps -- syncTraffic reads refs only
  }, [roads, traffic]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    junctionDataRef.current = { incidents, selectedId, junctions, onSelectIncident };
    if (junctions.length > LAYER_JUNCTION_THRESHOLD) {
      // Drop any DOM markers made before the full junction list arrived (the 5-junction fallback).
      markersRef.current.forEach((marker) => marker.remove());
      markersRef.current.clear();
      syncJunctionLayer();
      return;
    }
    const affected = affectedJunctionIds(incidents);
    const markerPoints =
      junctions.length > 0
        ? junctions.map((j) => ({ intersection_id: j.id, lon: j.lon, lat: j.lat }))
        : JUNCTIONS;

    for (const junction of markerPoints) {
      const incident = affected.get(junction.intersection_id);
      const isSelected = incident?.incident_id === selectedId;
      const color = incident ? (incident.pending_approval ? "#d9534f" : "#5cb85c") : "#337ab7";

      let marker = markersRef.current.get(junction.intersection_id);
      if (!marker) {
        const el = document.createElement("div");
        el.className = "junction-marker";
        marker = new Marker({ element: el })
          .setLngLat([junction.lon, junction.lat])
          .addTo(map);
        el.addEventListener("click", () => {
          const current = affectedJunctionIds(incidents).get(junction.intersection_id);
          if (current) onSelectIncident(current.incident_id);
        });
        markersRef.current.set(junction.intersection_id, marker);
      }
      const el = marker.getElement();
      el.style.background = color;
      el.style.width = isSelected ? "22px" : "16px";
      el.style.height = isSelected ? "22px" : "16px";
      el.title = incident
        ? `${junction.intersection_id}: ${incident.state.status}`
        : `${junction.intersection_id}: nominal`;
    }
  }, [incidents, selectedId, onSelectIncident, junctions]);

  return (
    <div className="map-wrap">
      <div ref={containerRef} className="map-view" />
      {showTrafficOverlay && <TrafficOverlay traffic={traffic} unavailable={trafficUnavailable} replay={replay} />}
    </div>
  );
}
