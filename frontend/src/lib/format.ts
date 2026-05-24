/**
 * Shared string formatters. Kept dependency-light so they work in both server
 * and client components (no Visx, no date-fns yet — pure ISO arithmetic).
 */

/** "2026-06-15T16:00:00+00:00" → "16:00 UTC". Returns null when malformed. */
export function utcTimeOfDay(iso: string | null | undefined): string | null {
  if (!iso) return null;
  // ISO 8601 timestamps have a 'T' separator and a 5-char HH:MM tail starting at index 11.
  if (iso.length < 16 || iso[10] !== "T") return null;
  return `${iso.slice(11, 16)} UTC`;
}

/** Format a probability in [0, 1] as a percentage. */
export function pct(p: number, digits = 1): string {
  return `${(p * 100).toFixed(digits)}%`;
}

/** Format with a leading sign (+0.42 / -0.18). */
export function signed(n: number, digits = 2): string {
  return `${n >= 0 ? "+" : ""}${n.toFixed(digits)}`;
}

/** "2026-06-11" → "Thu, Jun 11 2026". Returns the input on parse failure. */
export function prettyDate(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleDateString("en-US", {
    weekday: "short",
    month: "short",
    day: "numeric",
    year: "numeric",
    timeZone: "UTC",
  });
}
