import { notFound } from "next/navigation";

import { apiGet, ApiError, ApiUnreachable } from "@/lib/api";
import { ApiUnreachableBanner } from "@/components/api-unreachable-banner";
import { ForecastHeader } from "@/components/forecast-header";
import { VersusHeader } from "@/components/team-chip";
import { EloNarrative } from "@/components/match/elo-narrative";
import { H2HTable } from "@/components/match/h2h-table";
import { OutcomeTiles } from "@/components/match/outcome-tiles";
import { ProbHeatmap } from "@/components/match/prob-heatmap";
import { RecentFormBadges } from "@/components/match/recent-form-badges";
import { LiveSection } from "@/components/match/live-section";
import { DownloadableCard } from "@/components/downloadable-card";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { utcTimeOfDay } from "@/lib/format";
import type { FixtureWithPrediction } from "@/lib/types";

export default async function MatchDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const matchId = Number(id);
  if (Number.isNaN(matchId) || matchId < 0 || matchId > 71) {
    notFound();
  }

  let detail: FixtureWithPrediction | null = null;
  let unreachable = false;
  try {
    detail = await apiGet<FixtureWithPrediction>(
      `/api/v1/matches/${matchId}`,
      undefined,
      { revalidate: 300 },
    );
  } catch (err) {
    if (err instanceof ApiUnreachable) {
      unreachable = true;
    } else if (err instanceof ApiError && err.status === 404) {
      notFound();
    } else if (!(err instanceof ApiError)) {
      throw err;
    }
  }

  if (unreachable || !detail) {
    return (
      <div className="space-y-4">
        <h1 className="text-2xl font-semibold">Match detail</h1>
        <ApiUnreachableBanner />
      </div>
    );
  }

  const { fixture, prediction } = detail;
  const kickoff = utcTimeOfDay(fixture.utc_kickoff);
  const when = kickoff ? `${fixture.date} ${kickoff}` : fixture.date;

  return (
    <div className="space-y-6">
      <header className="space-y-2">
        <h1 className="text-2xl font-semibold tracking-tight">Match detail</h1>
        <ForecastHeader />
      </header>

      <section className="space-y-2">
        <VersusHeader home={fixture.home_team} away={fixture.away_team} />
        <p className="text-xs text-muted-foreground">
          Group {fixture.group} · {fixture.city}, {fixture.country} · {when} ·{" "}
          {fixture.neutral
            ? "neutral venue"
            : `${fixture.home_team} at home`}
        </p>
      </section>

      <OutcomeTiles
        matchId={matchId}
        homeTeam={fixture.home_team}
        awayTeam={fixture.away_team}
        probs={prediction.outcome}
      />

      {prediction.score_matrix ? (
        <DownloadableCard
          title="Joint score probability"
          filename={`heatmap-${fixture.home_team}-${fixture.away_team}`}
        >
          <ProbHeatmap
            matrix={prediction.score_matrix}
            homeTeam={fixture.home_team}
            awayTeam={fixture.away_team}
          />
        </DownloadableCard>
      ) : null}

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Why this prediction</CardTitle>
        </CardHeader>
        <CardContent>
          <EloNarrative
            homeTeam={fixture.home_team}
            awayTeam={fixture.away_team}
            neutral={fixture.neutral}
            expectedHomeGoals={prediction.expected_home_goals}
            expectedAwayGoals={prediction.expected_away_goals}
            topScoreline={prediction.top_scorelines[0]}
          />
        </CardContent>
      </Card>

      <LiveSection matchId={matchId} />

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">
              {fixture.home_team} — last 5
            </CardTitle>
          </CardHeader>
          <CardContent>
            <RecentFormBadges team={fixture.home_team} n={5} />
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle className="text-base">
              {fixture.away_team} — last 5
            </CardTitle>
          </CardHeader>
          <CardContent>
            <RecentFormBadges team={fixture.away_team} n={5} />
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Head-to-head</CardTitle>
        </CardHeader>
        <CardContent>
          <H2HTable homeTeam={fixture.home_team} awayTeam={fixture.away_team} n={10} />
        </CardContent>
      </Card>
    </div>
  );
}
