"use client";

import { useEffect, useRef, useState } from "react";

import { apiUrl } from "@/lib/api";

export type LiveSnapshot = {
  match_id: number;
  home_team: string;
  away_team: string;
  minute: number;
  period: number;
  home_score: number;
  away_score: number;
  home_red_cards: number;
  away_red_cards: number;
  last_event_type: string;
  win_prob: { home_win: number; draw: number; away_win: number };
  win_prob_source: string;
};

type State = {
  snapshot: LiveSnapshot | null;
  events: LiveSnapshot[];
  status: "idle" | "open" | "closed" | "error";
  error: string | null;
};

const FT_WHISTLE = "FT_WHISTLE";
const INITIAL_BACKOFF_MS = 1000;
const MAX_BACKOFF_MS = 30_000;

/**
 * SSE subscription for the Phase 6 live win-prob endpoint.
 *
 * - Opens an `EventSource` against the absolute API URL so traffic skips
 *   the Vercel edge entirely and hits Fly directly (no serverless timeout).
 * - Closes deterministically when a `FT_WHISTLE` arrives so finished matches
 *   don't leave dangling connections.
 * - On accidental disconnect (non-terminal `error`), reconnects with
 *   exponential backoff capped at 30s.
 *
 * Pass `enabled={false}` to skip subscribing entirely (e.g. when the API has
 * no events for this match yet — the page can probe `/live/{id}` first and
 * only flip the hook on when the response is non-pre-match).
 */
export function useLiveWinProb(matchId: number, opts?: { enabled?: boolean }) {
  const enabled = opts?.enabled ?? true;
  const [state, setState] = useState<State>({
    snapshot: null,
    events: [],
    status: "idle",
    error: null,
  });

  const sourceRef = useRef<EventSource | null>(null);
  const backoffRef = useRef<number>(INITIAL_BACKOFF_MS);
  const retryTimerRef = useRef<number | null>(null);
  const closedByFtRef = useRef<boolean>(false);

  useEffect(() => {
    if (!enabled) return;
    closedByFtRef.current = false;
    backoffRef.current = INITIAL_BACKOFF_MS;

    function close() {
      sourceRef.current?.close();
      sourceRef.current = null;
      if (retryTimerRef.current != null) {
        window.clearTimeout(retryTimerRef.current);
        retryTimerRef.current = null;
      }
    }

    function connect() {
      const url = apiUrl(`/api/v1/live/${matchId}/sse`);
      const es = new EventSource(url);
      sourceRef.current = es;
      es.onopen = () => {
        backoffRef.current = INITIAL_BACKOFF_MS;
        setState((s) => ({ ...s, status: "open", error: null }));
      };
      es.onmessage = (evt) => {
        try {
          const snap = JSON.parse(evt.data) as LiveSnapshot;
          setState((s) => ({
            ...s,
            snapshot: snap,
            events: [...s.events, snap],
            status: "open",
            error: null,
          }));
          if (snap.last_event_type === FT_WHISTLE) {
            closedByFtRef.current = true;
            close();
            setState((s) => ({ ...s, status: "closed" }));
          }
        } catch (err) {
          setState((s) => ({
            ...s,
            error: err instanceof Error ? err.message : "parse error",
          }));
        }
      };
      es.onerror = () => {
        // EventSource fires onerror on both transient blips and permanent
        // close; check the readyState to tell them apart.
        if (closedByFtRef.current) return;
        const closedByServer = es.readyState === EventSource.CLOSED;
        es.close();
        sourceRef.current = null;
        setState((s) => ({ ...s, status: closedByServer ? "closed" : "error" }));
        if (!closedByServer) {
          const delay = Math.min(backoffRef.current, MAX_BACKOFF_MS);
          backoffRef.current = Math.min(backoffRef.current * 2, MAX_BACKOFF_MS);
          retryTimerRef.current = window.setTimeout(connect, delay);
        }
      };
    }

    connect();
    return () => {
      close();
    };
  }, [matchId, enabled]);

  return state;
}
