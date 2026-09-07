import { useEffect, useRef } from "react";
import { Map as MapLibreMap, Marker } from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import corridorGeometry from "./corridor_geometry.json";
import type { Incident } from "./api";

// Real junction coordinates for the Week 5 corridor (manhattan_midtown, 5 contiguous
// signalized junctions), exported from the actual built SUMO network by
// scripts/export_corridor_geometry.py -- not hand-placed pins.
const JUNCTIONS: { intersection_id: string; lon: number; lat: number }[] =
  corridorGeometry.junctions;

interface MapViewProps {
  incidents: Incident[];
  selectedId: string | null;
  onSelectIncident: (incidentId: string) => void;
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

export function MapView({ incidents, selectedId, onSelectIncident }: MapViewProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MapLibreMap | null>(null);
  const markersRef = useRef<Map<string, Marker>>(new Map());

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;
    const center: [number, number] = [JUNCTIONS[0].lon, JUNCTIONS[0].lat];
    mapRef.current = new MapLibreMap({
      container: containerRef.current,
      // Free, no-API-key vector basemap -- appropriate for a read-only public demo
      // (docs/design/10_deployment.md: "zero control endpoints, no bus, no LLM keys").
      style: "https://demotiles.maplibre.org/style.json",
      center,
      zoom: 15,
    });
    return () => {
      mapRef.current?.remove();
      mapRef.current = null;
    };
  }, []);

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

  return <div ref={containerRef} className="map-view" />;
}
