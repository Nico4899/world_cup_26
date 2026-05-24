import { apiGet, ApiError, ApiUnreachable } from "@/lib/api";
import { ApiUnreachableBanner } from "@/components/api-unreachable-banner";
import { ForecastHeader } from "@/components/forecast-header";
import { MetricCard } from "@/components/metric-card";
import { TeamChip } from "@/components/team-chip";
import { DownloadableCard } from "@/components/downloadable-card";
import { EloLine } from "@/components/team/elo-line";
import { PathToFinal } from "@/components/team/path-to-final";
import { TeamPicker } from "@/components/team/team-picker";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { EmptyPanel } from "@/components/empty-panel";
import { HelpDot } from "@/components/help-dot";
import { RecentFormBadges } from "@/components/match/recent-form-badges";
import { pct, signed } from "@/lib/format";
import type { FixtureSummary } from "@/lib/types";

type EloHistory = {
  team: string;
  history: { snapshot_date: string; rating: number; global_rank: number | null }[];
};
type TeamProbs = {
  team: string;
  run_id: number | null;
  n_sims: number | null;
  model_version: string | null;
  group_winner_p: number | null;
  group_runner_up_p: number | null;
  advance_r32_p: number | null;
  advance_r16_p: number | null;
  quarterfinal_p: number | null;
  semifinal_p: number | null;
  final_p: number | null;
  champion_p: number | null;
};
type FifaRanks = {
  team: string;
  history: { ranking_date: string; rank: number; points: number | null; previous_rank: number | null }[];
};
type XgForm = {
  team: string;
  last_5: { matches: number; xg_for: number | null; xg_against: number | null; xg_diff: number | null };
  last_10: { matches: number; xg_for: number | null; xg_against: number | null; xg_diff: number | null };
  last_12_months: { matches: number; xg_for: number | null; xg_against: number | null; xg_diff: number | null };
};
type Squad = {
  team: string;
  tournament: string | null;
  snapshot_date: string | null;
  players: {
    player_name: string;
    shirt_number: number | null;
    position: string | null;
    birth_date: string | null;
    club: string | null;
    caps: number | null;
    goals: number | null;
  }[];
};
type PathToFinalResponse = {
  team: string;
  n_sims: number;
  rounds: {
    round: "r32" | "r16" | "qf" | "sf" | "final";
    p_reached: number;
    most_likely_opponent: { team: string; p_conditional: number } | null;
    top_opponents: { team: string; p_conditional: number }[];
  }[];
};

export default async function TeamProfilePage({
  params,
}: {
  params: Promise<{ name: string }>;
}) {
  const { name: rawName } = await params;
  const team = decodeURIComponent(rawName);

  let unreachable = false;
  let fixtures: FixtureSummary[] = [];
  let probs: TeamProbs | null = null;
  let elo: EloHistory | null = null;
  let ranks: FifaRanks | null = null;
  let xg: XgForm | null = null;
  let squad: Squad | null = null;
  let path: PathToFinalResponse | null = null;

  try {
    [fixtures, probs, elo, ranks, xg, squad, path] = await Promise.all([
      apiGet<FixtureSummary[]>("/api/v1/matches", undefined, { revalidate: 600 }),
      apiGet<TeamProbs>(`/api/v1/teams/${team}/tournament-probs`, undefined, {
        revalidate: 600,
      }).catch(() => null),
      apiGet<EloHistory>(`/api/v1/teams/${team}/elo-history`, undefined, {
        revalidate: 3600,
      }).catch(() => null),
      apiGet<FifaRanks>(`/api/v1/teams/${team}/fifa-rankings`, undefined, {
        revalidate: 3600,
      }).catch(() => null),
      apiGet<XgForm>(`/api/v1/teams/${team}/xg-form`, undefined, {
        revalidate: 1800,
      }).catch(() => null),
      apiGet<Squad>(`/api/v1/teams/${team}/squad`, undefined, {
        revalidate: 3600,
      }).catch(() => null),
      apiGet<PathToFinalResponse>(
        `/api/v1/teams/${team}/path-to-final`,
        { n_sims: 2000 },
        { revalidate: 1800 },
      ).catch(() => null),
    ]);
  } catch (err) {
    if (err instanceof ApiUnreachable) unreachable = true;
    else if (!(err instanceof ApiError)) throw err;
  }

  if (unreachable) {
    return (
      <div className="space-y-4">
        <h1 className="ds-h1">Team Profile</h1>
        <ApiUnreachableBanner />
      </div>
    );
  }

  const teams = Array.from(
    new Set([...(fixtures ?? []).flatMap((m) => [m.home_team, m.away_team])]),
  ).sort();

  const opponentByRound = new Map(
    (path?.rounds ?? []).map((r) => [r.round, r.most_likely_opponent] as const),
  );

  function roundPopover(round: "r16" | "qf" | "sf" | "final") {
    if (!probs?.run_id) return undefined;
    const opp = opponentByRound.get(round);
    return (
      <div className="space-y-1.5">
        <p className="text-xs">
          From persisted run <strong>#{probs.run_id}</strong> ({probs.n_sims} sims,
          model <code>{probs.model_version ?? "unknown"}</code>).
        </p>
        {opp ? (
          <p className="text-xs">
            Most-likely opponent at this stage: <strong>{opp.team}</strong>{" "}
            ({pct(opp.p_conditional, 0)} of paths).
          </p>
        ) : (
          <p className="text-xs italic">
            Most-likely opponent not yet observed in the path-to-final sample.
          </p>
        )}
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <header className="space-y-2">
        <h1 className="ds-h1">Team Profile</h1>
        <ForecastHeader />
      </header>

      <TeamPicker current={team} teams={teams} />

      <div className="text-2xl">
        <TeamChip team={team} size="lg" bold />
      </div>

      <section className="space-y-2">
        <h2 className="text-lg font-semibold">WC 2026 advancement probabilities</h2>
        {!probs || probs.run_id == null ? (
          <EmptyPanel
            title="No Monte Carlo run on disk yet"
            hint="Advancement odds will appear here once the daily refit + persist job has written a run."
            cta={{ href: "/groups", label: "See group-stage forecasts" }}
          />
        ) : probs.champion_p == null ? (
          <EmptyPanel
            title={`No row for ${team} in run #${probs.run_id}`}
            hint={`The latest run (${probs.n_sims} sims) didn't include this team — likely a team-list change since the last refit.`}
            cta={{ href: "/groups", label: "See group-stage forecasts" }}
          />
        ) : (
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            <MetricCard
              label="Champion"
              value={pct(probs.champion_p)}
              popoverTitle="Champion probability"
              popover={roundPopover("final")}
            />
            <MetricCard
              label="Final"
              value={pct(probs.final_p ?? 0)}
              popoverTitle="Reach final"
              popover={roundPopover("sf")}
            />
            <MetricCard
              label="Semifinal"
              value={pct(probs.semifinal_p ?? 0)}
              popoverTitle="Reach SF"
              popover={roundPopover("qf")}
            />
            <MetricCard
              label="Quarterfinal"
              value={pct(probs.quarterfinal_p ?? 0)}
              popoverTitle="Reach QF"
              popover={roundPopover("r16")}
            />
          </div>
        )}
      </section>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Path to the final</CardTitle>
        </CardHeader>
        <CardContent>
          {path ? (
            <PathToFinal rounds={path.rounds} nSims={path.n_sims} />
          ) : (
            <EmptyPanel
              title="Path-to-final not available"
              hint="The model couldn't sample 2,000 knockout paths for this team — usually because the persisted Monte Carlo run is empty or this team isn't in the bracket."
            />
          )}
        </CardContent>
      </Card>

      {elo && elo.history.length > 0 ? (
        <DownloadableCard
          title={
            <span className="inline-flex items-center gap-1">
              Elo rating history
              <HelpDot term="Elo" />
            </span>
          }
          filename={`elo-history-${team}`}
        >
          <EloLine history={elo.history} />
          <p className="text-xs text-muted-foreground mt-2">
            {elo.history.length} daily snapshots. Most recent:{" "}
            {elo.history[elo.history.length - 1].snapshot_date} →{" "}
            {elo.history[elo.history.length - 1].rating.toFixed(1)}
          </p>
        </DownloadableCard>
      ) : (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Elo rating history</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-xs text-muted-foreground italic">
              No Elo snapshots on disk for this team.
            </p>
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader>
          <CardTitle className="text-base">
            Recent form (last 10 internationals)
          </CardTitle>
        </CardHeader>
        <CardContent>
          <RecentFormBadges team={team} n={10} />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">FIFA Men&apos;s Ranking history</CardTitle>
        </CardHeader>
        <CardContent>
          {ranks && ranks.history.length > 0 ? (
            <FifaSummary history={ranks.history} />
          ) : (
            <p className="text-xs text-muted-foreground italic">
              No FIFA ranking snapshots on file yet.
            </p>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base inline-flex items-center gap-1">
            xG form
            <HelpDot term="xG" />
          </CardTitle>
        </CardHeader>
        <CardContent>
          {!xg || (xg.last_10.matches === 0 && xg.last_12_months.matches === 0) ? (
            <p className="text-xs text-muted-foreground italic">
              No xG records for this team yet.
            </p>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <XgPanel label="Last 5" split={xg.last_5} />
              <XgPanel label="Last 10" split={xg.last_10} />
              <XgPanel label="Last 12 months" split={xg.last_12_months} />
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Squad</CardTitle>
        </CardHeader>
        <CardContent>
          {!squad || squad.players.length === 0 ? (
            <p className="text-xs text-muted-foreground italic">
              No squad snapshot on file. Run the manual{" "}
              <code>wikipedia_squads_refresh</code> job from the Operator page once
              squads are announced.
            </p>
          ) : (
            <>
              <p className="text-xs text-muted-foreground mb-2">
                {squad.players.length} players · snapshot{" "}
                {squad.snapshot_date ?? "unknown date"} ({squad.tournament ?? "—"}).
              </p>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>#</TableHead>
                    <TableHead>Player</TableHead>
                    <TableHead>Pos</TableHead>
                    <TableHead>Club</TableHead>
                    <TableHead>Born</TableHead>
                    <TableHead className="text-right">Caps</TableHead>
                    <TableHead className="text-right">Goals</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {squad.players.map((p) => (
                    <TableRow key={p.player_name}>
                      <TableCell className="text-xs tabular-nums">
                        {p.shirt_number ?? "—"}
                      </TableCell>
                      <TableCell className="text-xs">{p.player_name}</TableCell>
                      <TableCell className="text-xs">{p.position ?? "—"}</TableCell>
                      <TableCell className="text-xs">{p.club ?? "—"}</TableCell>
                      <TableCell className="text-xs">{p.birth_date ?? "—"}</TableCell>
                      <TableCell className="text-xs text-right tabular-nums">
                        {p.caps ?? "—"}
                      </TableCell>
                      <TableCell className="text-xs text-right tabular-nums">
                        {p.goals ?? "—"}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function FifaSummary({ history }: { history: FifaRanks["history"] }) {
  const latest = history[history.length - 1];
  const delta =
    latest.previous_rank != null ? latest.previous_rank - latest.rank : null;
  return (
    <div className="space-y-3">
      <div className="grid grid-cols-3 gap-3">
        <MetricCard label="Current rank" value={`#${latest.rank}`} />
        <MetricCard
          label="Points"
          value={latest.points != null ? latest.points.toFixed(0) : "—"}
        />
        <MetricCard
          label={<>Δ since prev. snapshot <HelpDot term="delta" /></>}
          value={delta == null ? "—" : signed(delta, 0)}
          help="Positive = climbed (rank went down)."
        />
      </div>
    </div>
  );
}

function XgPanel({
  label,
  split,
}: {
  label: string;
  split: XgForm["last_5"];
}) {
  const n = split.matches ?? 0;
  return (
    <div className="space-y-1">
      <p className="text-sm font-medium">
        {label} — {n} match{n === 1 ? "" : "es"}
      </p>
      {n === 0 ? (
        <p className="text-xs text-muted-foreground italic">no rows in window</p>
      ) : (
        <div className="grid grid-cols-3 gap-1.5">
          <Stat
            label="xG for / match"
            value={split.xg_for != null ? split.xg_for.toFixed(2) : "—"}
          />
          <Stat
            label="xG against / match"
            value={split.xg_against != null ? split.xg_against.toFixed(2) : "—"}
          />
          <Stat
            label="xG diff"
            value={split.xg_diff != null ? signed(split.xg_diff) : "—"}
          />
        </div>
      )}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border p-2">
      <p className="text-[10px] text-muted-foreground uppercase">{label}</p>
      <p className="text-sm font-medium tabular-nums">{value}</p>
    </div>
  );
}
