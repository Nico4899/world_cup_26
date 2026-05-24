import Link from "next/link";
import { ArrowRight } from "lucide-react";

import { VersusHeader } from "@/components/team-chip";
import { Card, CardContent } from "@/components/ui/card";
import { buttonVariants } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import {
  ProbabilityBar,
  ProbabilityLegend,
  type ProbabilitySegment,
} from "@/components/probability-bar";
import { HelpDot } from "@/components/help-dot";
import { WhyPopover } from "@/components/today/why-popover";
import { apiGet, ApiError, ApiUnreachable } from "@/lib/api";
import { utcTimeOfDay } from "@/lib/format";
import type { FixtureSummary, FixtureWithPrediction } from "@/lib/types";

const COLORS = {
  home: "var(--outcome-home)",
  draw: "var(--outcome-draw)",
  away: "var(--outcome-away)",
} as const;

export async function MatchCard({ fixture }: { fixture: FixtureSummary }) {
  let detail: FixtureWithPrediction | null = null;
  try {
    detail = await apiGet<FixtureWithPrediction>(
      `/api/v1/matches/${fixture.match_id}`,
      undefined,
      { revalidate: 300 },
    );
  } catch (err) {
    // 404 (out-of-range) shouldn't happen here since /matches already
    // filtered to valid ids; surface anything else as a placeholder.
    if (!(err instanceof ApiUnreachable) && !(err instanceof ApiError)) {
      throw err;
    }
  }

  const kickoff = utcTimeOfDay(fixture.utc_kickoff);
  const caption = [
    kickoff ? `⏱ ${kickoff}` : null,
    `Group ${fixture.group}`,
    `${fixture.city}, ${fixture.country}`,
    fixture.neutral ? "neutral" : "home advantage",
  ]
    .filter(Boolean)
    .join(" · ");

  if (!detail) {
    return (
      <Card variant="ribbon">
        <CardContent className="space-y-2 py-4">
          <VersusHeader home={fixture.home_team} away={fixture.away_team} />
          <p className="text-xs text-muted-foreground">{caption}</p>
          <p className="text-xs text-destructive">
            Prediction unavailable. The API is unreachable.
          </p>
        </CardContent>
      </Card>
    );
  }

  const pred = detail.prediction;
  const segments: ProbabilitySegment[] = [
    { label: fixture.home_team, value: pred.outcome.home_win, color: COLORS.home },
    { label: "Draw", value: pred.outcome.draw, color: COLORS.draw },
    { label: fixture.away_team, value: pred.outcome.away_win, color: COLORS.away },
  ];

  return (
    <Card variant="ribbon">
      <CardContent className="space-y-3 py-4">
        <VersusHeader home={fixture.home_team} away={fixture.away_team} />
        <p className="text-xs text-muted-foreground">{caption}</p>
        <ProbabilityBar segments={segments} height={32} />
        <ProbabilityLegend segments={segments} />
        <div className="flex items-center justify-between text-sm pt-1">
          <div>
            <p className="text-xs uppercase tracking-wide text-muted-foreground inline-flex items-center gap-1">
              Expected goals
              <HelpDot term="xG" />
            </p>
            <p className="font-medium tabular-nums">
              {pred.expected_home_goals.toFixed(2)} - {pred.expected_away_goals.toFixed(2)}
            </p>
          </div>
          <div className="text-right">
            <p className="text-xs uppercase tracking-wide text-muted-foreground">
              Most likely
            </p>
            <ul className="text-xs space-y-0.5">
              {pred.top_scorelines.slice(0, 3).map((sc, i) => (
                <li key={i}>
                  <strong>
                    {sc.home_goals}-{sc.away_goals}
                  </strong>{" "}
                  ({(sc.probability * 100).toFixed(1)}%)
                </li>
              ))}
            </ul>
          </div>
        </div>
        <div className="grid grid-cols-2 gap-2">
          <WhyPopover
            matchId={fixture.match_id}
            homeTeam={fixture.home_team}
            awayTeam={fixture.away_team}
            expectedHomeGoals={pred.expected_home_goals}
            expectedAwayGoals={pred.expected_away_goals}
            topScoreline={pred.top_scorelines[0]}
          />
          <Link
            href={`/match/${fixture.match_id}`}
            className={cn(buttonVariants({ variant: "outline", size: "sm" }), "w-full")}
          >
            Open detail
            <ArrowRight className="h-3.5 w-3.5 ml-1" aria-hidden />
          </Link>
        </div>
      </CardContent>
    </Card>
  );
}
