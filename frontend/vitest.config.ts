import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import { resolve } from "node:path";

/**
 * Unit-test config separate from Next's build.
 *
 * - `jsdom` for DOM globals so React Testing Library can mount components.
 * - `setupFiles` registers @testing-library/jest-dom matchers globally.
 * - Excludes the Playwright e2e directory so `pnpm test` is fast.
 */
export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./vitest.setup.ts"],
    include: ["src/**/*.test.{ts,tsx}"],
    exclude: ["node_modules", ".next", "e2e"],
  },
  resolve: {
    alias: {
      "@": resolve(__dirname, "src"),
    },
  },
});
