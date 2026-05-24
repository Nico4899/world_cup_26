import { apiGet } from "@/lib/api";
import type { HealthResponse, WC2026TrackRecord } from "@/lib/types";

function formatUtc(iso: string | null): string | null {
  if (!iso) return null;
  const cleaned = iso.endsWith("Z") ? iso.slice(0, -1) + "+00:00" : iso;
  const d = new Date(cleaned);
  if (Number.isNaN(d.getTime())) return null;
  // YYYY-MM-DD HH:MM UTC
  const pad = (n: number) => n.toString().padStart(2, "0");
  return (
    `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())} ` +
    `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())} UTC`
  );
}

function lastCompletedSummary(tr: WC2026TrackRecord | null): string | null {
  if (!tr || !tr.per_match || tr.per_match.length === 0) return null;
  // Endpoint isn't ordered explicitly; pick the latest match_date row.
  const latest = tr.per_match.reduce((acc, row) =>
    !acc || row.match_date > acc.match_date ? row : acc,
  );
  return `${latest.home_team} ${latest.home_score}-${latest.away_score} ${latest.away_team}`;
}

/**
 * "As of … after X-Y" freshness banner shown atop every forecast page.
 *
 * Server Component — fetches both endpoints at request time on the Vercel
 * edge / Node runtime, never ships the fetch logic to the browser. Silent
 * degrade: if either endpoint fails, the banner renders nothing rather
 * than alarming the user (the page-level fetch will surface its own warning).
 */
export async function ForecastHeader() {
  let health: HealthResponse | null = null;
  let track: WC2026TrackRecord | null = null;
  try {
    [health, track] = await Promise.all([
      apiGet<HealthResponse>("/health", undefined, { noStore: true }),
      apiGet<WC2026TrackRecord>("/api/v1/track-record/wc2026", undefined, {
        revalidate: 60,
      }).catch(() => null),
    ]);
  } catch {
    return null;
  }
  if (!health) return null;

  const fitAt = formatUtc(health.model_fit_at);
  if (!fitAt) return null;
  const after = lastCompletedSummary(track);
  const modelVersion = health.model_version ?? "model";

  return (
    <p className="text-xs text-muted-foreground italic">
      As of <strong className="not-italic">{fitAt}</strong>
      {after ? <>, after {after}</> : null}
      <span className="mx-1.5">·</span>
      <span className="not-italic">{modelVersion}</span>
    </p>
  );
}
