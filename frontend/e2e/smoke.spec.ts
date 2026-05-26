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

test("top nav lists 7 primary tabs + Operator icon", async ({ page }) => {
  await page.goto("/");
  // 7 high-frequency tabs render their full label.
  for (const label of [
    "Today",
    "Match Detail",
    "Groups",
    "Bracket",
    "Track Record",
    "Map",
    "About",
  ]) {
    await expect(page.getByRole("link", { name: label })).toBeVisible();
  }
  // Operator sits in the chrome as an icon-only link (it's intentionally
  // technical so it doesn't crowd the primary tabs).
  await expect(page.getByRole("link", { name: "Operator" })).toBeVisible();
});

test("command palette reaches every route", async ({ page }) => {
  await page.goto("/");
  // Dismiss the first-visit tour if it shows so the click below isn't intercepted.
  await page.evaluate(() => window.localStorage.setItem("wc2026-onboarded", "1"));
  await page.reload();
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

test("first-visit tour opens once, then localStorage gates it", async ({ page }) => {
  // Clear the flag, visit /, expect the modal.
  await page.goto("/");
  await page.evaluate(() => window.localStorage.removeItem("wc2026-onboarded"));
  await page.reload();
  const tour = page.getByRole("dialog", { name: /each card is one match/i });
  await expect(tour).toBeVisible({ timeout: 10_000 });
  // Skip dismisses + persists the flag.
  await tour.getByRole("button", { name: /skip tour/i }).click();
  await expect(tour).toBeHidden();
  // Reload — flag is set, tour does NOT reappear.
  await page.reload();
  await expect(page.getByRole("dialog", { name: /each card is one match/i })).toBeHidden();
});

test("mobile hamburger sheet exposes 7 primary tabs + Operator", async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 720 });
  await page.goto("/");
  // Persist the tour flag so it doesn't block the hamburger.
  await page.evaluate(() => window.localStorage.setItem("wc2026-onboarded", "1"));
  await page.reload();
  await page.getByRole("button", { name: /open primary navigation/i }).click();
  const sheet = page.locator("[data-slot='sheet-content']");
  await expect(sheet).toBeVisible();
  for (const label of [
    "Today",
    "Match Detail",
    "Groups",
    "Bracket",
    "Track Record",
    "Map",
    "About",
    "Operator",
  ]) {
    await expect(sheet.getByRole("link", { name: label })).toBeVisible();
  }
});
