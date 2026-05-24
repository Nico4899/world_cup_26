import { apiGet, ApiError, ApiUnreachable } from "@/lib/api";
import { ApiUnreachableBanner } from "@/components/api-unreachable-banner";
import { ForecastHeader } from "@/components/forecast-header";
import { MetricCard } from "@/components/metric-card";
import { DownloadableCard } from "@/components/downloadable-card";
import {
  ReliabilityScatter,
  type ReliabilityBin,
} from "@/components/track-record/reliability-scatter";
import { TournamentTabs } from "@/components/track-record/tournament-tabs";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { HelpDot } from "@/components/help-dot";
import { pct, signed } from "@/lib/format";
import type { WC2026TrackRecord } from "@/lib/types";

export const metadata = { title: "Track Record — WC 2026 Predictions" };

type Tournament = "WC2022" | "WC2018";

type HistoricalHeadline = {
  n_matches: number;
  log_loss: number;
  brier: number;
  rps: number;
  baseline_log_loss: number;
  base_h: number;
  base_d: number;
  base_a: number;
};

type HistoricalResponse = {
  tournament: Tournament;
  headline: HistoricalHeadline;
  reliability: ReliabilityBin[];
  bookmaker_reference: {
    log_loss_low: number;
    log_loss_high: number;
    cite: string;
  } | null;
};

const fmtMetric = (n: number | null) => (n == null ? "—" : n.toFixed(4));

export default async function TrackRecordPage({
  searchParams,
}: {
  searchParams: Promise<{ tournament?: string }>;
}) {
  const params = await searchParams;
  const tournament: Tournament = params.tournament === "WC2018" ? "WC2018" : "WC2022";

  let tr: WC2026TrackRecord | null = null;
  let historical: HistoricalResponse | null = null;
  let unreachable = false;
  try {
    [tr, historical] = await Promise.all([
      apiGet<WC2026TrackRecord>("/api/v1/track-record/wc2026", undefined, {
        revalidate: 120,
      }),
      apiGet<HistoricalResponse>(
        `/api/v1/track-record/historical/${tournament}`,
        undefined,
        { revalidate: 86_400 },
      ).catch(() => null),
    ]);
  } catch (err) {
    if (err instanceof ApiUnreachable) unreachable = true;
    else if (!(err instanceof ApiError)) throw err;
  }

  return (
    <div className="space-y-6">
      <header className="space-y-2">
        <h1 className="ds-h1">Track record</h1>
        <ForecastHeader />
        <p className="text-xs text-muted-foreground">
          How close our past predictions were to reality. Lower numbers are
          better — a coin-flip baseline scores a{" "}
          <HelpDot term="log-loss" /> of 1.099; perfect prediction is 0.
          The reliability diagram below shows whether the model is well{" "}
          <HelpDot term="calibration" />: dots on the dashed line mean
          predicted probabilities matched observed frequencies.
        </p>
      </header>

      {unreachable ? <ApiUnreachableBanner /> : null}

      {tr ? (
        <section className="space-y-3">
          <h2 className="text-lg font-semibold">WC 2026 — live rolling calibration</h2>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            <MetricCard
              label="Completed matches"
              value={String(tr.n_completed)}
              help="WC 2026 fixtures with a final score on file."
            />
            <MetricCard
              label={<>Log-loss <HelpDot term="log-loss" /></>}
              value={fmtMetric(tr.log_loss)}
              popoverTitle="Log-loss (negative log-likelihood)"
              popover={
                <>
                  <p>
                    Penalises confident wrong predictions more than cautious
                    ones. A coin-flip baseline scores ~1.099; perfect
                    prediction scores 0.
                  </p>
                  <p>
                    Updated after each completed WC 2026 fixture; based on
                    the model&apos;s pre-match probabilities (no look-ahead).
                  </p>
                </>
              }
            />
            <MetricCard
              label={<>Brier <HelpDot term="Brier" /></>}
              value={fmtMetric(tr.brier)}
              popoverTitle="Brier score"
              popover={
                <p>
                  Squared error between predicted and observed
                  probabilities for each outcome. Lower is better. The
                  climatological baseline for 1X2 is around 0.22.
                </p>
              }
            />
            <MetricCard
              label={<>RPS <HelpDot term="RPS" /></>}
              value={fmtMetric(tr.rps)}
              popoverTitle="Ranked Probability Score"
              popover={
                <p>
                  Penalises predictions that miss the order of outcomes
                  more than ones that are close — predicting a draw when
                  the home team won hurts less than predicting away.
                </p>
              }
            />
          </div>
          {tr.per_match.length > 0 ? (
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
          ) : (
            <p className="text-sm text-muted-foreground italic">
              No completed WC 2026 matches recorded yet — the live event poller
              writes here once a fixture has a FT_WHISTLE row.
            </p>
          )}
        </section>
      ) : null}

      <section className="space-y-3">
        <div className="flex items-center justify-between flex-wrap gap-3">
          <h2 className="text-lg font-semibold inline-flex items-center gap-1">
            Historical hindcasts
            <HelpDot term="hindcast" />
          </h2>
          <TournamentTabs current={tournament} />
        </div>
        {historical ? (
          <div className="space-y-3">
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
              <MetricCard
                label="Matches"
                value={String(historical.headline.n_matches)}
              />
              <MetricCard
                label={<>Log-loss <HelpDot term="log-loss" /></>}
                value={historical.headline.log_loss.toFixed(4)}
                delta={`${signed(historical.headline.log_loss - historical.headline.baseline_log_loss, 4)} vs base rates`}
              />
              <MetricCard
                label={<>Brier <HelpDot term="Brier" /></>}
                value={historical.headline.brier.toFixed(4)}
              />
              <MetricCard
                label={<>RPS <HelpDot term="RPS" /></>}
                value={historical.headline.rps.toFixed(4)}
              />
            </div>
            <p className="text-xs text-muted-foreground">
              Base rates (climatological no-skill model): H={pct(historical.headline.base_h)},
              D={pct(historical.headline.base_d)}, A={pct(historical.headline.base_a)}.
              Climatological log-loss:{" "}
              {historical.headline.baseline_log_loss.toFixed(4)}. Lower is
              better. Negative delta = we beat the no-skill baseline.
            </p>
            {historical.bookmaker_reference ? (
              <div className="rounded-md border bg-card p-3">
                <p className="text-sm">
                  <strong>Bookmaker reference</strong> (closing-odds-implied
                  probabilities): log-loss ≈{" "}
                  <strong>
                    {historical.bookmaker_reference.log_loss_low.toFixed(2)}–
                    {historical.bookmaker_reference.log_loss_high.toFixed(2)}
                  </strong>
                  . Source: <em>{historical.bookmaker_reference.cite}</em>.
                  Bookmaker odds incorporate injuries, news, and market signals
                  our model cannot see — a 0.04-0.08 gap is the realistic ceiling.
                </p>
              </div>
            ) : null}
            <DownloadableCard
              title="Reliability diagram"
              filename={`reliability-${tournament.toLowerCase()}`}
            >
              <ReliabilityScatter bins={historical.reliability} />
            </DownloadableCard>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Outcome</TableHead>
                  <TableHead className="text-right">Bin low</TableHead>
                  <TableHead className="text-right">Bin high</TableHead>
                  <TableHead className="text-right">n</TableHead>
                  <TableHead className="text-right">Predicted</TableHead>
                  <TableHead className="text-right">Realised</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {historical.reliability.map((b, i) => (
                  <TableRow key={i}>
                    <TableCell className="text-xs">{b.outcome}</TableCell>
                    <TableCell className="text-xs text-right tabular-nums">
                      {pct(b.bin_low, 0)}
                    </TableCell>
                    <TableCell className="text-xs text-right tabular-nums">
                      {pct(b.bin_high, 0)}
                    </TableCell>
                    <TableCell className="text-xs text-right tabular-nums">
                      {b.n}
                    </TableCell>
                    <TableCell className="text-xs text-right tabular-nums">
                      {(b.mean_predicted * 100).toFixed(1)}%
                    </TableCell>
                    <TableCell className="text-xs text-right tabular-nums">
                      {(b.realized_frequency * 100).toFixed(1)}%
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        ) : (
          <p className="text-sm text-muted-foreground italic">
            Historical hindcast unavailable. The first request takes ~3s
            server-side; if it consistently fails, check the API logs.
          </p>
        )}
      </section>
    </div>
  );
}
