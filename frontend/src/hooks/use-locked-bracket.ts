"use client";

import { useEffect, useState } from "react";

export type Lock = { match_id: number; winner: string };

const STORAGE_KEY = "wc2026.bracket_locks";

function read(): Lock[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(
      (x): x is Lock =>
        x &&
        typeof x === "object" &&
        typeof x.match_id === "number" &&
        typeof x.winner === "string",
    );
  } catch {
    return [];
  }
}

/**
 * localStorage-backed lock list for the Bracket page's conditional-MC mode.
 *
 * - Replaces Streamlit's `st.session_state["bracket_locks"]`.
 * - Survives page refresh and tab restore (unlike the Streamlit version).
 * - Adding a lock for a match_id that already has one overwrites it, so the
 *   form can be reused to change a previously-picked winner.
 *
 * The initial render returns `[]` (so SSR/CSR hydration matches), then
 * effect-loads the stored value on the first client paint.
 */
export function useLockedBracket(): {
  locks: Lock[];
  add: (lock: Lock) => void;
  remove: (matchId: number) => void;
  clear: () => void;
  hydrated: boolean;
} {
  const [locks, setLocks] = useState<Lock[]>([]);
  const [hydrated, setHydrated] = useState(false);

  // Intentional set-state-in-effect: localStorage is unavailable during SSR,
  // so the first render returns `[]` and the effect synchronously hydrates
  // from storage after mount. Switching to useSyncExternalStore is possible
  // but adds a `getServerSnapshot` no-op that doesn't buy anything here.
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setLocks(read());
    setHydrated(true);
  }, []);

  useEffect(() => {
    if (!hydrated) return;
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(locks));
    } catch {
      // Storage quota / sandboxed iframes — silently skip; the in-memory
      // copy still drives the UI for the session.
    }
  }, [locks, hydrated]);

  return {
    locks,
    hydrated,
    add: (lock) =>
      setLocks((prev) => [
        ...prev.filter((l) => l.match_id !== lock.match_id),
        lock,
      ]),
    remove: (matchId) =>
      setLocks((prev) => prev.filter((l) => l.match_id !== matchId)),
    clear: () => setLocks([]),
  };
}
