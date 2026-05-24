/**
 * Curated host-city coordinates for FIFA WC 2026. Ships with the repo so the
 * map page works without an external geocoding API. Country colours:
 * USA navy, Mexico green, Canada red.
 */
export type HostCity = {
  city: string;
  lat: number;
  lon: number;
  country: "United States" | "Mexico" | "Canada";
};

export const HOST_CITIES: HostCity[] = [
  // United States (11 venues)
  { city: "Atlanta", lat: 33.7553, lon: -84.4006, country: "United States" },
  { city: "Boston", lat: 42.0909, lon: -71.2643, country: "United States" },
  { city: "Dallas", lat: 32.7473, lon: -97.0945, country: "United States" },
  { city: "Houston", lat: 29.6847, lon: -95.4107, country: "United States" },
  { city: "Kansas City", lat: 39.0489, lon: -94.4839, country: "United States" },
  { city: "Los Angeles", lat: 33.9535, lon: -118.3392, country: "United States" },
  { city: "Miami", lat: 25.958, lon: -80.2389, country: "United States" },
  {
    city: "New York/New Jersey",
    lat: 40.8135,
    lon: -74.0745,
    country: "United States",
  },
  { city: "Philadelphia", lat: 39.9008, lon: -75.1675, country: "United States" },
  {
    city: "San Francisco Bay Area",
    lat: 37.403,
    lon: -121.97,
    country: "United States",
  },
  { city: "Seattle", lat: 47.5952, lon: -122.3316, country: "United States" },
  // Mexico (3 venues)
  { city: "Mexico City", lat: 19.3029, lon: -99.1505, country: "Mexico" },
  { city: "Guadalajara", lat: 20.6818, lon: -103.4626, country: "Mexico" },
  { city: "Monterrey", lat: 25.6691, lon: -100.2453, country: "Mexico" },
  // Canada (2 venues)
  { city: "Toronto", lat: 43.6332, lon: -79.4196, country: "Canada" },
  { city: "Vancouver", lat: 49.2767, lon: -123.1119, country: "Canada" },
];

export const COUNTRY_FILL: Record<HostCity["country"], [number, number, number]> = {
  "United States": [31, 119, 180], // navy
  Mexico: [44, 160, 44], // green
  Canada: [214, 39, 40], // red
};
