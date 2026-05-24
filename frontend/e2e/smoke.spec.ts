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

test("sidebar nav lists all 9 routes", async ({ page }) => {
  await page.goto("/");
  // Sidebar is hidden on small viewports; Playwright's Desktop Chrome
  // default is >= lg breakpoint so the nav should be visible.
  for (const label of [
    "Today",
    "Match Detail",
    "Groups",
    "Bracket",
    "Track Record",
    "About",
    "Operator",
    "Team Profile",
    "Map",
  ]) {
    await expect(page.getByRole("link", { name: label })).toBeVisible();
  }
});
