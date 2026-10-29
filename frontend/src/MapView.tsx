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
import type { Incident, TrafficEdgeGeometry, TrafficSnapshot } from "./api";
import { TrafficOverlay } from "./TrafficOverlay";

// Real junction coordinates for the Week 5 corridor (manhattan_midtown, 5 contiguous
// signalized junctions), exported from the actual built SUMO network by
// scripts/export_corridor_geometry.py -- not hand-placed pins.
const JUNCTIONS: { intersection_id: string; lon: number; lat: number }[] =
  corridorGeometry.junctions;

const ROADS_SOURCE = "roads";
const VEHICLES_SOURCE = "vehicles";

// Road colour = how close traffic is to the speed limit (mean speed / limit):
// grey = empty road, red < 35%, amber 35-65%, green >= 65%.
const ROAD_COLOR: ExpressionSpecification = [
  "case",
  ["<", ["get", "ratio"], 0],
  "#8d99a6",
  ["step", ["get", "ratio"], "#d9534f", 0.35, "#f0ad4e", 0.65, "#2ecc71"],
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

function buildRoadFeatures(
  roads: TrafficEdgeGeometry[],
  traffic: TrafficSnapshot | null,
): FeatureCollectionLike {
  const disrupted = new Set((traffic?.disruptions ?? []).map((d) => d.edge_id));
  return {
    type: "FeatureCollection",
    features: roads.map((road) => ({
      type: "Feature",
      properties: {
        edge_id: road.edge_id,
        // -1 = no vehicle on the road right now (drawn grey)
        ratio: traffic?.edges[road.edge_id]?.speed_ratio ?? -1,
        disrupted: disrupted.has(road.edge_id) ? 1 : 0,
      },
      geometry: { type: "LineString", coordinates: road.coords },
    })),
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
  traffic: TrafficSnapshot | null;
  /** True when the API has no live twin (or can't be reached). */
  trafficUnavailable: boolean;
  /** False in the static public demo build, which has no live backend at all. */
  showTrafficOverlay: boolean;
}

export function MapView({
  incidents,
  selectedId,
  onSelectIncident,
  roads,
  traffic,
  trafficUnavailable,
  showTrafficOverlay,
}: MapViewProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MapLibreMap | null>(null);
  const markersRef = useRef<Map<string, Marker>>(new Map());
  const styleLoadedRef = useRef(false);
  const trafficDataRef = useRef({ roads, traffic });

  // Pushes the latest roads/vehicles into the map. Safe to call at any time: it does
  // nothing until the basemap style has loaded, and the style's `style.load` handler calls it
  // again so data that arrived first is not lost.
  function syncTraffic() {
    const map = mapRef.current;
    if (!map || !styleLoadedRef.current) return;
    const { roads: currentRoads, traffic: currentTraffic } = trafficDataRef.current;
    if (currentRoads.length === 0) return;

    if (!map.getSource(ROADS_SOURCE)) {
      map.addSource(ROADS_SOURCE, { type: "geojson", data: EMPTY_COLLECTION as GeoJsonData });
      map.addSource(VEHICLES_SOURCE, { type: "geojson", data: EMPTY_COLLECTION as GeoJsonData });
      map.addLayer({
        id: "roads-disruption",
        type: "line",
        source: ROADS_SOURCE,
        filter: ["==", ["get", "disrupted"], 1],
        layout: { "line-cap": "round" },
        paint: { "line-color": "#7b2cbf", "line-opacity": 0.55, "line-width": 14 },
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

    (map.getSource(ROADS_SOURCE) as GeoJSONSource).setData(
      buildRoadFeatures(currentRoads, currentTraffic) as GeoJsonData,
    );
    (map.getSource(VEHICLES_SOURCE) as GeoJSONSource).setData(
      buildVehicleFeatures(currentTraffic) as GeoJsonData,
    );
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
    // `style.load`, not `load`: `load` also waits for the first basemap tiles, which
    // would hold the traffic layers back (or hide them for good if tiles are blocked).
    map.on("style.load", () => {
      if (mapRef.current !== map) return;
      styleLoadedRef.current = true;
      syncTraffic();
    });
    return () => {
      // Markers belong to this map instance. Drop them with it so a remount (React
      // StrictMode runs effects twice in dev) re-creates them on the new map instead
      // of leaving them attached to the one that was just removed.
      markersRef.current.forEach((marker) => marker.remove());
      markersRef.current.clear();
      styleLoadedRef.current = false;
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
    const affected = affectedJunctionIds(incidents);

    for (const junction of JUNCTIONS) {
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
  }, [incidents, selectedId, onSelectIncident]);

  return (
    <div className="map-wrap">
      <div ref={containerRef} className="map-view" />
      {showTrafficOverlay && <TrafficOverlay traffic={traffic} unavailable={trafficUnavailable} />}
    </div>
  );
}
