import { Suspense } from "react";

import { apiGet, ApiError, ApiUnreachable } from "@/lib/api";
import { ApiUnreachableBanner } from "@/components/api-unreachable-banner";
import { ForecastHeader } from "@/components/forecast-header";
import { DatePicker } from "@/components/today/date-picker";
import { GroupStrip } from "@/components/today/group-strip";
import { MatchCard } from "@/components/today/match-card";
import type { FixtureSummary } from "@/lib/types";

const DEFAULT_DATE = "2026-06-11";

/**
 * Today's predictions — server-rendered, ?date= controls the matchday.
 * Each match card fetches its own detailed prediction in parallel via
 * Next.js's per-fetch memoisation, so re-rendering on date change reuses
 * the cached snapshot when it's still fresh.
 */
export default async function TodayPage({
  searchParams,
}: {
  searchParams: Promise<{ date?: string }>;
}) {
  const params = await searchParams;
  const date = params.date ?? DEFAULT_DATE;

  let matches: FixtureSummary[] = [];
  let unreachable = false;
  try {
    matches = await apiGet<FixtureSummary[]>(
      "/api/v1/matches",
      { date },
      { revalidate: 300 },
    );
  } catch (err) {
    if (err instanceof ApiUnreachable) {
      unreachable = true;
    } else if (!(err instanceof ApiError)) {
      throw err;
    }
  }

  return (
    <div className="space-y-6">
      <header className="space-y-2">
        <h1 className="ds-h1">Today&apos;s predictions</h1>
        <ForecastHeader />
      </header>

      <DatePicker initial={date} />

      {unreachable ? (
        <ApiUnreachableBanner />
      ) : matches.length === 0 ? (
        <p className="text-sm text-muted-foreground">
          No matches scheduled on {date}.
        </p>
      ) : (
        <>
          <p className="text-xs text-muted-foreground">
            {matches.length} match{matches.length === 1 ? "" : "es"} on {date}
          </p>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {matches.map((m) => (
              <MatchCard key={m.match_id} fixture={m} />
            ))}
          </div>
        </>
      )}

      <Suspense fallback={null}>
        <GroupStrip />
      </Suspense>
    </div>
  );
}
