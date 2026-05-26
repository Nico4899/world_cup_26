/**
 * Next.js App-Router handler for NextAuth.js v5.
 *
 * NextAuth handles its own routing under `/api/auth/...` — the catch-all
 * here just re-exports the GET / POST handlers from our config module.
 * When auth is disabled (missing env vars), both methods return 503 with
 * a JSON error so the dashboard can show a "sign-in disabled" hint.
 */
export { GET, POST } from "@/lib/auth-route";
