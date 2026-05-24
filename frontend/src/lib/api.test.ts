import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

import { apiGet, apiPost, ApiError, ApiUnreachable } from "./api";

describe("apiGet", () => {
  const originalFetch = global.fetch;
  afterEach(() => {
    global.fetch = originalFetch;
  });

  it("returns the parsed JSON body on 2xx", async () => {
    global.fetch = vi.fn(async () =>
      new Response(JSON.stringify({ status: "ok" }), { status: 200 }),
    );
    const result = await apiGet<{ status: string }>("/health");
    expect(result).toEqual({ status: "ok" });
  });

  it("appends query params with truthy values", async () => {
    const seen: { url: string }[] = [];
    global.fetch = vi.fn(async (input: URL | RequestInfo) => {
      seen.push({ url: String(input) });
      return new Response(JSON.stringify([]), { status: 200 });
    });
    await apiGet("/api/v1/matches", { date: "2026-06-11", group: "A", skip: undefined });
    expect(seen[0].url).toContain("date=2026-06-11");
    expect(seen[0].url).toContain("group=A");
    expect(seen[0].url).not.toContain("skip="); // undefined values are dropped
  });

  it("throws ApiError on non-2xx, with status + parsed body", async () => {
    global.fetch = vi.fn(async () =>
      new Response(JSON.stringify({ detail: "no XGB" }), { status: 503 }),
    );
    await expect(apiGet("/api/v1/explain/0")).rejects.toMatchObject({
      status: 503,
      body: { detail: "no XGB" },
    });
    await expect(apiGet("/api/v1/explain/0")).rejects.toBeInstanceOf(ApiError);
  });

  it("throws ApiUnreachable when fetch itself fails", async () => {
    global.fetch = vi.fn(async () => {
      throw new TypeError("Failed to fetch");
    });
    await expect(apiGet("/health")).rejects.toBeInstanceOf(ApiUnreachable);
  });

  it("returns undefined on 204 No Content", async () => {
    global.fetch = vi.fn(async () => new Response(null, { status: 204 }));
    expect(await apiGet("/anything")).toBeUndefined();
  });
});

describe("apiPost", () => {
  const originalFetch = global.fetch;
  beforeEach(() => {
    global.fetch = vi.fn(async (input: URL | RequestInfo, init?: RequestInit) => {
      return new Response(
        JSON.stringify({ method: init?.method, body: init?.body, url: String(input) }),
        { status: 200 },
      );
    });
  });
  afterEach(() => {
    global.fetch = originalFetch;
  });

  it("sends the body as JSON with content-type", async () => {
    const result = (await apiPost("/api/v1/_ops/run-job/foo", { x: 1 })) as {
      method: string;
      body: string;
    };
    expect(result.method).toBe("POST");
    expect(JSON.parse(result.body)).toEqual({ x: 1 });
  });
});
