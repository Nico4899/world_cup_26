"use client";

import { useState } from "react";
import { DeckGL } from "@deck.gl/react";
import { ScatterplotLayer } from "@deck.gl/layers";
import { Map as MapLibre } from "react-map-gl/maplibre";
import "maplibre-gl/dist/maplibre-gl.css";

import { HOST_CITIES, COUNTRY_FILL, type HostCity } from "@/lib/host-cities";

const INITIAL_VIEW = {
  latitude: 35.0,
  longitude: -100.0,
  zoom: 2.6,
  pitch: 0,
  bearing: 0,
};

/**
 * 16-pin host-city map. Uses `react-map-gl/maplibre` for the basemap
 * (no API key required; uses the open OSM/CARTO tiles) and `@deck.gl/react`'s
 * ScatterplotLayer for the pins with per-country fill + tooltip on hover.
 */
export function HostCityMap() {
  const [hovered, setHovered] = useState<HostCity | null>(null);

  const layer = new ScatterplotLayer<HostCity>({
    id: "host-cities",
    data: HOST_CITIES,
    getPosition: (d) => [d.lon, d.lat],
    getFillColor: (d) => COUNTRY_FILL[d.country],
    getRadius: 30_000,
    radiusUnits: "meters",
    stroked: true,
    getLineColor: [255, 255, 255],
    lineWidthMinPixels: 1,
    opacity: 0.85,
    pickable: true,
    onHover: ({ object }) => setHovered((object as HostCity | null) ?? null),
  });

  return (
    <div className="relative w-full h-[420px] rounded-lg overflow-hidden border">
      <DeckGL
        initialViewState={INITIAL_VIEW}
        controller
        layers={[layer]}
        style={{ position: "absolute", top: "0", left: "0", right: "0", bottom: "0" }}
      >
        <MapLibre
          mapStyle="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
          attributionControl={{ compact: true }}
        />
      </DeckGL>
      {hovered ? (
        <div className="pointer-events-none absolute top-2 left-2 z-10 rounded-md bg-background/95 px-3 py-1.5 text-xs shadow ring-1 ring-border">
          <strong>{hovered.city}</strong>{" "}
          <span className="text-muted-foreground">({hovered.country})</span>
        </div>
      ) : null}
    </div>
  );
}
