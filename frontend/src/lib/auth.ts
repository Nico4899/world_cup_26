/**
 * NextAuth.js v5 (Auth.js) configuration for the WC 2026 dashboard.
 *
 * Magic-link sign-in only (no OAuth providers). Sessions are stored in
 * Postgres via `@auth/pg-adapter`, sharing the same database the Python
 * backend already uses (the `users` / `accounts` / `sessions` /
 * `verification_tokens` tables are created by Alembic migration
 * `a4c0e1f2b3d4`).
 *
 * Graceful degradation
 * --------------------
 * The auth handlers must never block the build or the dev server. When
 * any of the required env vars is missing — `AUTH_SECRET`,
 * `DATABASE_URL`, `RESEND_API_KEY` — `authEnabled` is false and the
 * exported handlers return 503. The rest of the app (predictions,
 * track-record, /map, etc.) keeps working untouched.
 *
 * Env vars
 * --------
 * - `AUTH_SECRET`      session-cookie signing key (`openssl rand -base64 32`).
 * - `DATABASE_URL`     Postgres URL. May carry SQLAlchemy's `postgresql+psycopg`
 *                      scheme — we normalise it for the Node `pg` driver.
 * - `RESEND_API_KEY`   Resend account API key (3000 free emails/month).
 * - `EMAIL_FROM`       The "From:" address on the magic-link email.
 */

import type { NextRequest } from "next/server";

const AUTH_SECRET = process.env.AUTH_SECRET ?? "";
const DATABASE_URL = process.env.DATABASE_URL ?? "";
const RESEND_API_KEY = process.env.RESEND_API_KEY ?? "";
const EMAIL_FROM = process.env.EMAIL_FROM ?? "";

/**
 * `true` exactly when every required env var is present. Callers may read
 * this to short-circuit UI (e.g. hide the Sign-in button when auth is off).
 */
export const authEnabled: boolean = Boolean(
  AUTH_SECRET && DATABASE_URL && RESEND_API_KEY && EMAIL_FROM,
);

/**
 * Normalise SQLAlchemy's `postgresql+psycopg://…` to the plain
 * `postgresql://…` the Node `pg` driver expects.
 */
function pgConnectionString(raw: string): string {
  return raw.replace(/^postgresql\+psycopg(2|ng)?:\/\//, "postgresql://");
}

type Handler = (req: NextRequest) => Promise<Response>;

interface AuthExports {
  handlers: { GET: Handler; POST: Handler };
  auth: () => Promise<{ user?: { id: string; email: string } } | null>;
  signIn: (...args: unknown[]) => Promise<Response>;
  signOut: (...args: unknown[]) => Promise<Response>;
}

/**
 * Lazy NextAuth instantiation — keeps the build green when the auth env
 * vars are unset (CI, local dev without a Postgres + Resend). The first
 * actual `/api/auth/...` request triggers the real init; if it fails,
 * subsequent requests fall back to the 503 stub.
 */
let cachedExports: AuthExports | null = null;
let initFailed = false;

async function buildExports(): Promise<AuthExports | null> {
  if (!authEnabled || initFailed) return null;
  if (cachedExports) return cachedExports;
  try {
    const [{ default: NextAuth }, { default: PostgresAdapter }, { default: Resend }, { Pool }] =
      await Promise.all([
        import("next-auth"),
        import("@auth/pg-adapter"),
        import("next-auth/providers/resend"),
        import("pg"),
      ]);
    const pool = new Pool({ connectionString: pgConnectionString(DATABASE_URL) });
    const config = {
      adapter: PostgresAdapter(pool),
      providers: [
        Resend({
          apiKey: RESEND_API_KEY,
          from: EMAIL_FROM,
        }),
      ],
      secret: AUTH_SECRET,
      session: { strategy: "database" as const },
    };
    // NextAuth v5's `NextAuth(config)` returns `{ auth, handlers, signIn, signOut }`.
    const result = NextAuth(config) as unknown as AuthExports;
    cachedExports = result;
    return result;
  } catch (err) {
    console.error("auth: NextAuth init failed; serving 503 from /api/auth/*", err);
    initFailed = true;
    return null;
  }
}

function stubHandler(): Handler {
  return async () =>
    new Response(
      JSON.stringify({
        error:
          "auth not configured — set AUTH_SECRET, DATABASE_URL, RESEND_API_KEY, EMAIL_FROM",
      }),
      { status: 503, headers: { "Content-Type": "application/json" } },
    );
}

/**
 * Exported handlers for the `[...nextauth]` route. We wrap NextAuth's
 * handlers so the import-time path stays cheap (no Postgres pool created
 * until the first auth request arrives).
 */
export const handlers = {
  GET: async (req: NextRequest) => {
    const real = await buildExports();
    return real ? real.handlers.GET(req) : stubHandler()(req);
  },
  POST: async (req: NextRequest) => {
    const real = await buildExports();
    return real ? real.handlers.POST(req) : stubHandler()(req);
  },
};

/**
 * Server-component helper. Returns the current session payload, or `null`
 * when the user is unauthenticated OR auth is disabled.
 */
export async function auth(): Promise<
  { user?: { id: string; email: string } } | null
> {
  const real = await buildExports();
  return real ? real.auth() : null;
}
