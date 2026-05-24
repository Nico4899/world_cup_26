import { defineConfig, devices } from "@playwright/test";

const PORT = Number(process.env.PORT ?? 3100);
const BASE_URL = process.env.WC2026_FRONTEND_URL ?? `http://localhost:${PORT}`;

/**
 * Playwright smoke configuration.
 *
 * Spins up `next start` against a pre-built bundle on PORT so the tests run
 * against a production-ish build. The API URL is taken from
 * `NEXT_PUBLIC_API_URL` if set; otherwise the smoke tests assert only on
 * page chrome (header, nav, ApiUnreachableBanner) so CI without a live
 * backend still passes.
 */
export default defineConfig({
  testDir: "./e2e",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 2 : undefined,
  reporter: process.env.CI ? "github" : "list",
  use: {
    baseURL: BASE_URL,
    trace: "retain-on-failure",
  },
  projects: [
    { name: "chromium", use: { ...devices["Desktop Chrome"] } },
  ],
  webServer: process.env.WC2026_FRONTEND_URL
    ? undefined
    : {
        command: `next build && next start -p ${PORT}`,
        url: BASE_URL,
        timeout: 240_000,
        reuseExistingServer: !process.env.CI,
      },
});
