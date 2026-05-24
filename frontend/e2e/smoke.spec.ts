import { test, expect } from "@playwright/test";

/**
 * Cross-route smoke test: each of the 9 dashboard routes must render its
 * chrome + a recognisable page header even when the FastAPI backend is
 * unreachable.
 *
 * When `NEXT_PUBLIC_API_URL` points at a live backend (set on staging),
 * Playwright also asserts that at least one `<svg>` chart is present per
 * data-driven page — useful as a "chart actually rendered" regression net.
 */

const ROUTES: { path: string; heading: string | RegExp; charts?: boolean }[] = [
  { path: "/", heading: /today's predictions/i, charts: true },
  { path: "/match/0", heading: /match detail/i, charts: true },
  { path: "/groups", heading: /group-stage advancement/i, charts: true },
  { path: "/bracket", heading: /knockout bracket realisation/i },
  { path: "/track-record", heading: /track record/i, charts: true },
  { path: "/about", heading: /about \/ methodology/i },
  { path: "/ops", heading: /operator/i },
  { path: "/team/Argentina", heading: /team profile/i, charts: true },
  { path: "/map", heading: /host-city map/i },
];

for (const { path, heading, charts } of ROUTES) {
  test(`smoke: ${path}`, async ({ page }) => {
    await page.goto(path);
    await expect(page.getByRole("heading", { name: heading })).toBeVisible({
      timeout: 15_000,
    });
    if (charts && process.env.NEXT_PUBLIC_API_URL) {
      // Allow ample time for server-rendered pages to fetch + render charts.
      await expect(page.locator("svg").first()).toBeVisible({ timeout: 30_000 });
    }
  });
}

test("top nav lists 5 primary tabs", async ({ page }) => {
  await page.goto("/");
  // Top-nav has the 5 high-frequency routes; the other 4 utility routes
  // are reachable via the ⌘K command palette (see next test).
  for (const label of ["Today", "Match Detail", "Groups", "Bracket", "Track Record"]) {
    await expect(page.getByRole("link", { name: label })).toBeVisible();
  }
});

test("command palette reaches every route", async ({ page }) => {
  await page.goto("/");
  // Open the palette via the visible "Jump to…" trigger (works on any OS).
  await page.getByRole("button", { name: /open command palette/i }).click();
  // The palette renders all 9 route labels under the "Routes" group.
  const palette = page.getByRole("dialog", { name: /command palette/i });
  await expect(palette).toBeVisible();
  for (const label of [
    "Today",
    "Match Detail",
    "Groups",
    "Bracket",
    "Track Record",
    "Team Profile",
    "Map",
    "About",
    "Operator",
  ]) {
    await expect(palette.getByRole("button", { name: new RegExp(`^${label}\\b`) })).toBeVisible();
  }
});
