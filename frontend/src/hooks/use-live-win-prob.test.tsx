import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { renderHook, act, waitFor } from "@testing-library/react";

import { useLiveWinProb, type LiveSnapshot } from "./use-live-win-prob";

// EventSource isn't implemented in jsdom; we stub the global with a
// controllable mock that tests drive via `fakeSource.emit(...)`.
class FakeEventSource {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSED = 2;
  static instances: FakeEventSource[] = [];

  readonly CONNECTING = 0;
  readonly OPEN = 1;
  readonly CLOSED = 2;

  url: string;
  readyState: number = FakeEventSource.OPEN;
  onopen: ((evt: Event) => void) | null = null;
  onmessage: ((evt: MessageEvent) => void) | null = null;
  onerror: ((evt: Event) => void) | null = null;
  closed = false;

  constructor(url: string) {
    this.url = url;
    FakeEventSource.instances.push(this);
    queueMicrotask(() => this.onopen?.(new Event("open")));
  }

  emit(payload: LiveSnapshot) {
    this.onmessage?.(new MessageEvent("message", { data: JSON.stringify(payload) }));
  }

  fireError(closeReason: "transient" | "permanent" = "transient") {
    if (closeReason === "permanent") this.readyState = FakeEventSource.CLOSED;
    this.onerror?.(new Event("error"));
  }

  close() {
    this.closed = true;
    this.readyState = FakeEventSource.CLOSED;
  }
}

const baseSnap = (over: Partial<LiveSnapshot> = {}): LiveSnapshot => ({
  match_id: 5,
  home_team: "Argentina",
  away_team: "Spain",
  minute: 12,
  period: 1,
  home_score: 0,
  away_score: 0,
  home_red_cards: 0,
  away_red_cards: 0,
  last_event_type: "KICKOFF",
  win_prob: { home_win: 0.5, draw: 0.25, away_win: 0.25 },
  win_prob_source: "live_win_prob",
  ...over,
});

describe("useLiveWinProb", () => {
  const realEventSource = globalThis.EventSource;
  beforeEach(() => {
    FakeEventSource.instances = [];
    // @ts-expect-error swap global for the test
    globalThis.EventSource = FakeEventSource;
  });
  afterEach(() => {
    globalThis.EventSource = realEventSource;
  });

  it("opens a connection and tracks status transitions", async () => {
    const { result } = renderHook(() => useLiveWinProb(5));
    await waitFor(() => expect(result.current.status).toBe("open"));
    expect(FakeEventSource.instances).toHaveLength(1);
    expect(FakeEventSource.instances[0].url).toContain("/api/v1/live/5/sse");
  });

  it("appends snapshots from message events", async () => {
    const { result } = renderHook(() => useLiveWinProb(5));
    await waitFor(() => expect(result.current.status).toBe("open"));
    act(() => FakeEventSource.instances[0].emit(baseSnap({ minute: 12 })));
    act(() => FakeEventSource.instances[0].emit(baseSnap({ minute: 42 })));
    expect(result.current.events).toHaveLength(2);
    expect(result.current.snapshot?.minute).toBe(42);
  });

  it("closes the stream on FT_WHISTLE and surfaces status=closed", async () => {
    const { result } = renderHook(() => useLiveWinProb(5));
    await waitFor(() => expect(result.current.status).toBe("open"));
    act(() =>
      FakeEventSource.instances[0].emit(
        baseSnap({ minute: 90, last_event_type: "FT_WHISTLE" }),
      ),
    );
    await waitFor(() => expect(result.current.status).toBe("closed"));
    expect(FakeEventSource.instances[0].closed).toBe(true);
  });

  it("does not subscribe when enabled is false", () => {
    renderHook(() => useLiveWinProb(5, { enabled: false }));
    expect(FakeEventSource.instances).toHaveLength(0);
  });
});
