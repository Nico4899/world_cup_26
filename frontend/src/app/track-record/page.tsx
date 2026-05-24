import { apiGet, ApiError, ApiUnreachable } from "@/lib/api";
import { ApiUnreachableBanner } from "@/components/api-unreachable-banner";
import { ForecastHeader } from "@/components/forecast-header";
import { MetricCard } from "@/components/metric-card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { WC2026TrackRecord } from "@/lib/types";

export const metadata = { title: "Track Record — WC 2026 Predictions" };

const fmtMetric = (n: number | null) => (n == null ? "—" : n.toFixed(4));

/**
 * Track Record page — live WC 2026 rolling calibration above the (deferred)
 * historical hindcasts. Phase E adds a new `/api/v1/track-record/historical/
 * {tournament}` route and a reliability scatter below; until then this page
 * surfaces only the live metrics + per-match table.
 */
export default async function TrackRecordPage() {
  let tr: WC2026TrackRecord | null = null;
  let unreachable = false;
  try {
    tr = await apiGet<WC2026TrackRecord>("/api/v1/track-record/wc2026", undefined, {
      revalidate: 120,
    });
  } catch (err) {
    if (err instanceof ApiUnreachable) unreachable = true;
    else if (!(err instanceof ApiError)) throw err;
  }

  return (
    <div className="space-y-6">
      <header className="space-y-2">
        <h1 className="text-2xl font-semibold tracking-tight">Track record</h1>
        <ForecastHeader />
        <p className="text-xs text-muted-foreground">
          Calibration check on completed World Cups (out-of-sample). The live
          WC 2026 panel below grows as matches finish; the historical hindcasts
          (WC 2018 / WC 2022) land in Phase E.
        </p>
      </header>

      {unreachable ? (
        <ApiUnreachableBanner />
      ) : tr ? (
        <>
          <section className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            <MetricCard label="Completed matches" value={String(tr.n_completed)} />
            <MetricCard label="Log-loss" value={fmtMetric(tr.log_loss)} />
            <MetricCard label="Brier" value={fmtMetric(tr.brier)} />
            <MetricCard label="RPS" value={fmtMetric(tr.rps)} />
          </section>

          {tr.per_match.length > 0 ? (
            <section className="space-y-2">
              <h2 className="text-lg font-semibold">Per-match diagnostics</h2>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Date</TableHead>
                    <TableHead>Match</TableHead>
                    <TableHead>Score</TableHead>
                    <TableHead className="text-right">P(H)</TableHead>
                    <TableHead className="text-right">P(D)</TableHead>
                    <TableHead className="text-right">P(A)</TableHead>
                    <TableHead className="text-right">Log-loss</TableHead>
                    <TableHead className="text-right">Brier</TableHead>
                    <TableHead className="text-right">RPS</TableHead>
                    <TableHead>Model</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {tr.per_match.map((m, i) => (
                    <TableRow key={i}>
                      <TableCell className="text-xs">{m.match_date}</TableCell>
                      <TableCell className="text-xs">
                        {m.home_team} vs {m.away_team}
                      </TableCell>
                      <TableCell className="text-xs tabular-nums">
                        {m.home_score}-{m.away_score} ({m.observed})
                      </TableCell>
                      <TableCell className="text-xs text-right tabular-nums">
                        {(m.p_home * 100).toFixed(0)}%
                      </TableCell>
                      <TableCell className="text-xs text-right tabular-nums">
                        {(m.p_draw * 100).toFixed(0)}%
                      </TableCell>
                      <TableCell className="text-xs text-right tabular-nums">
                        {(m.p_away * 100).toFixed(0)}%
                      </TableCell>
                      <TableCell className="text-xs text-right tabular-nums">
                        {m.log_loss.toFixed(3)}
                      </TableCell>
                      <TableCell className="text-xs text-right tabular-nums">
                        {m.brier.toFixed(3)}
                      </TableCell>
                      <TableCell className="text-xs text-right tabular-nums">
                        {m.rps.toFixed(3)}
                      </TableCell>
                      <TableCell className="text-xs text-muted-foreground">
                        {m.model_version}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </section>
          ) : (
            <p className="text-sm text-muted-foreground">
              No completed WC 2026 matches recorded yet — the live event poller
              writes here once a fixture has a FT_WHISTLE row.
            </p>
          )}
        </>
      ) : null}

      <section className="space-y-2">
        <h2 className="text-lg font-semibold">Historical hindcasts</h2>
        <p className="text-sm text-muted-foreground">
          WC 2018 / WC 2022 day-by-day hindcast metrics + reliability diagrams
          land in Phase E (the API exposes them via a new{" "}
          <code>/api/v1/track-record/historical/&#123;tournament&#125;</code>{" "}
          endpoint).
        </p>
      </section>
    </div>
  );
}
