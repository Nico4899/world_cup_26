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

const SELECTED_BORDER: [number, number, number] = [250, 204, 21]; // amber-400

type Props = {
  /** Currently-selected city ("(all)" or one of HOST_CITIES.city). */
  selected?: string | null;
  /** Click a pin → emit its city, or `null` if the same pin was clicked twice. */
  onSelect?: (city: string | null) => void;
};

/**
 * 16-pin host-city map. Uses `react-map-gl/maplibre` for the basemap
 * (no API key required; uses the open OSM/CARTO tiles) and `@deck.gl/react`'s
 * ScatterplotLayer for the pins with per-country fill + tooltip on hover.
 *
 * When `onSelect` is provided, clicking a pin toggles the city filter on
 * the page (clicking the already-selected pin clears the filter).
 */
export function HostCityMap({ selected, onSelect }: Props) {
  const [hovered, setHovered] = useState<HostCity | null>(null);

  const layer = new ScatterplotLayer<HostCity>({
    id: "host-cities",
    data: HOST_CITIES,
    getPosition: (d) => [d.lon, d.lat],
    getFillColor: (d) => COUNTRY_FILL[d.country],
    getRadius: (d) => (d.city === selected ? 50_000 : 30_000),
    radiusUnits: "meters",
    stroked: true,
    getLineColor: (d) => (d.city === selected ? SELECTED_BORDER : [255, 255, 255]),
    getLineWidth: (d) => (d.city === selected ? 3 : 1),
    lineWidthUnits: "pixels",
    lineWidthMinPixels: 1,
    opacity: 0.85,
    pickable: true,
    onHover: ({ object }) => setHovered((object as HostCity | null) ?? null),
    onClick: ({ object }) => {
      if (!onSelect) return;
      const clicked = (object as HostCity | null) ?? null;
      if (!clicked) return;
      onSelect(clicked.city === selected ? null : clicked.city);
    },
    // Recompute the selected-pin styling whenever the prop flips so the
    // amber ring + larger radius appear immediately on click.
    updateTriggers: {
      getRadius: [selected],
      getLineColor: [selected],
      getLineWidth: [selected],
    },
  });

  return (
    <div className="relative w-full h-105 rounded-lg overflow-hidden border">
      <DeckGL
        initialViewState={INITIAL_VIEW}
        controller
        layers={[layer]}
        style={{ position: "absolute", top: "0", left: "0", right: "0", bottom: "0" }}
        getCursor={({ isHovering }) => (isHovering && onSelect ? "pointer" : "grab")}
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
          {onSelect ? (
            <span className="text-muted-foreground">
              {" "}· {hovered.city === selected ? "click to clear" : "click to filter"}
            </span>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
