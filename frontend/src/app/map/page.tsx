"use client";

import { useState } from "react";

import { HostCityMap } from "@/components/map/host-city-map";
import { ALL_CITIES, FixtureList } from "@/components/map/fixture-list";

/**
 * /map — clicking a pin on the deck.gl map filters the fixture table below,
 * and changing the dropdown highlights the corresponding pin. State lives at
 * the page level so the two views stay in sync.
 */
export default function MapPage() {
  const [selected, setSelected] = useState<string>(ALL_CITIES);

  return (
    <div className="space-y-6">
      <header className="space-y-2">
        <h1 className="ds-h1">Host-city map</h1>
        <p className="text-xs text-muted-foreground">
          16 host venues across USA (11) + Mexico (3) + Canada (2).
          Click a pin to filter the fixture list, or pick a city from the
          dropdown to highlight the pin.
        </p>
      </header>
      <HostCityMap
        selected={selected === ALL_CITIES ? null : selected}
        onSelect={(city) => setSelected(city ?? ALL_CITIES)}
      />
      <FixtureList selected={selected} onSelect={setSelected} />
    </div>
  );
}
