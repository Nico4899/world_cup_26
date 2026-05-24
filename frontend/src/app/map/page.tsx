import { HostCityMap } from "@/components/map/host-city-map";
import { FixtureList } from "@/components/map/fixture-list";

export const metadata = { title: "Host-city map — WC 2026 Predictions" };

export default function MapPage() {
  return (
    <div className="space-y-6">
      <header className="space-y-2">
        <h1 className="text-2xl font-semibold tracking-tight">Host-city map</h1>
        <p className="text-xs text-muted-foreground">
          16 host venues across USA (11) + Mexico (3) + Canada (2). Pick a city
          below to filter the fixture list to matches at that venue.
        </p>
      </header>
      <HostCityMap />
      <FixtureList />
    </div>
  );
}
