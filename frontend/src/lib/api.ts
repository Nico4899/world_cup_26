/**
 * Thin typed fetch wrapper for the WC 2026 FastAPI.
 *
 * - Reads the API base URL from `NEXT_PUBLIC_API_URL` (defaults to localhost
 *   for `pnpm dev`).
 * - Normalises HTTP errors into `ApiError` so React components can react to
 *   503/404 without re-parsing the response body.
 * - Adds a `noStore: true` flag so Server Components can opt out of Next's
 *   default fetch memoisation when freshness matters (e.g. /health).
 */

export const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  status: number;
  body: unknown;
  url: string;

  constructor(status: number, body: unknown, url: string) {
    super(`API ${status} at ${url}`);
    this.status = status;
    this.body = body;
    this.url = url;
  }
}

export class ApiUnreachable extends Error {
  url: string;

  constructor(url: string, cause?: unknown) {
    super(`Could not reach API at ${url}`);
    this.url = url;
    this.cause = cause;
  }
}

type FetchOptions = {
  /** Skip Next.js fetch memoisation/cache. */
  noStore?: boolean;
  /** Revalidate cached responses every N seconds. */
  revalidate?: number;
  /** Override the base URL. */
  baseUrl?: string;
  /** Outgoing headers. */
  headers?: HeadersInit;
  /** Abort signal. */
  signal?: AbortSignal;
};

function buildUrl(
  path: string,
  params: Record<string, string | number | boolean | undefined> | undefined,
  baseUrl: string,
) {
  const url = new URL(path, baseUrl);
  if (params) {
    for (const [key, value] of Object.entries(params)) {
      if (value === undefined) continue;
      url.searchParams.set(key, String(value));
    }
  }
  return url.toString();
}

async function request<T>(
  method: "GET" | "POST",
  path: string,
  {
    params,
    body,
    options,
  }: {
    params?: Record<string, string | number | boolean | undefined>;
    body?: unknown;
    options?: FetchOptions;
  } = {},
): Promise<T> {
  const baseUrl = options?.baseUrl ?? API_URL;
  const url = buildUrl(path, params, baseUrl);

  const next =
    options?.noStore === true
      ? { revalidate: 0 as const }
      : options?.revalidate !== undefined
        ? { revalidate: options.revalidate }
        : undefined;

  let res: Response;
  try {
    res = await fetch(url, {
      method,
      headers: {
        "Content-Type": "application/json",
        ...(options?.headers ?? {}),
      },
      body: body !== undefined ? JSON.stringify(body) : undefined,
      signal: options?.signal,
      ...(next ? { next } : { cache: "no-store" }),
    });
  } catch (err) {
    throw new ApiUnreachable(url, err);
  }

  if (!res.ok) {
    let payload: unknown = null;
    try {
      payload = await res.json();
    } catch {
      // empty body
    }
    throw new ApiError(res.status, payload, url);
  }

  // 204 No Content
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

export function apiGet<T>(
  path: string,
  params?: Record<string, string | number | boolean | undefined>,
  options?: FetchOptions,
): Promise<T> {
  return request<T>("GET", path, { params, options });
}

export function apiPost<T>(
  path: string,
  body: unknown,
  options?: FetchOptions,
): Promise<T> {
  return request<T>("POST", path, { body, options });
}

/** Build an absolute URL the browser can open (e.g. for EventSource). */
export function apiUrl(path: string): string {
  return buildUrl(path, undefined, API_URL);
}
