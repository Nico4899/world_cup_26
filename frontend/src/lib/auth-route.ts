/**
 * Re-export shim for the `[...nextauth]` App-Router route file.
 *
 * Next.js expects `GET` / `POST` named exports on the route module;
 * `lib/auth.ts` exposes them under `handlers.GET` / `handlers.POST` so
 * the rest of the app can still treat the auth surface as a single
 * object. This file is the one-line bridge.
 */

import { handlers } from "@/lib/auth";

export const GET = handlers.GET;
export const POST = handlers.POST;
