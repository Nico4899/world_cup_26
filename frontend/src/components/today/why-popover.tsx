"use client";

import { useQuery } from "@tanstack/react-query";
import { HelpCircle } from "lucide-react";

import { ApiError, apiGet } from "@/lib/api";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { TermHelp } from "@/components/help-dot";
import { signed, pct } from "@/lib/format";

type Contribution = { feature: string; value: number | null; contribution: number };
type Explanation = {
  contributions: Contribution[];
};

type Props = {
  matchId: number;
  homeTeam: string;
  awayTeam: string;
  expectedHomeGoals: number;
  expectedAwayGoals: number;
  topScoreline: { home_goals: number; away_goals: number; probability: number };
};

/**
 * Inline "Why?" drawer for the Today page match cards. Mirrors the spec's
 * "Why link opening the explanation drawer" requirement: clicking opens a
 * popover with the xG narrative + top scoreline + best-effort SHAP top-3.
 *
 * SHAP fetches lazily on open; silently falls back to "no model contributions"
 * when XGB isn't loaded (503 from /explain).
 */
export function WhyPopover({
  matchId,
  homeTeam,
  awayTeam,
  expectedHomeGoals,
  expectedAwayGoals,
  topScoreline,
}: Props) {
  return (
    <Popover>
      <PopoverTrigger
        className="inline-flex items-center justify-center gap-1.5 rounded-md border bg-background px-2.5 py-1.5 text-xs font-medium hover:bg-accent w-full"
      >
        <HelpCircle className="h-3.5 w-3.5" aria-hidden />
        Why?
      </PopoverTrigger>
      <PopoverContent className="w-80">
        <WhyBody
          matchId={matchId}
          homeTeam={homeTeam}
          awayTeam={awayTeam}
          expectedHomeGoals={expectedHomeGoals}
          expectedAwayGoals={expectedAwayGoals}
          topScoreline={topScoreline}
        />
      </PopoverContent>
    </Popover>
  );
}

function WhyBody({
  matchId,
  homeTeam,
  awayTeam,
  expectedHomeGoals,
  expectedAwayGoals,
  topScoreline,
}: Props) {
  const xgDiff = expectedHomeGoals - expectedAwayGoals;
  const favourite = xgDiff >= 0 ? homeTeam : awayTeam;
  const edge =
    Math.abs(xgDiff) < 0.2
      ? "roughly even"
      : `edge ${favourite} ${signed(Math.abs(xgDiff))}`;
  return (
    <div className="space-y-2 text-sm">
      <p>
        <span className="font-medium">
          <TermHelp term="xG">Expected goals</TermHelp>:
        </span>{" "}
        {homeTeam} <strong>{expectedHomeGoals.toFixed(2)}</strong> vs {awayTeam}{" "}
        <strong>{expectedAwayGoals.toFixed(2)}</strong> ({edge}).
      </p>
      <p>
        Most likely scoreline:{" "}
        <strong>
          {topScoreline.home_goals}-{topScoreline.away_goals}
        </strong>{" "}
        ({pct(topScoreline.probability)}).
      </p>
      <ShapTop3 matchId={matchId} />
    </div>
  );
}

function ShapTop3({ matchId }: { matchId: number }) {
  const { data, error } = useQuery({
    queryKey: ["explain", matchId, "home_win", 3],
    queryFn: () =>
      apiGet<Explanation>(`/api/v1/explain/${matchId}`, {
        class_name: "home_win",
        top_n: 3,
      }),
    staleTime: 5 * 60 * 1000,
    retry: false,
  });
  if (error) {
    const status = error instanceof ApiError ? error.status : null;
    if (status === 503) {
      return (
        <p className="text-xs text-muted-foreground">
          SHAP contributions require the XGB artefact (not loaded server-side).
        </p>
      );
    }
    return null;
  }
  const contribs = data?.contributions ?? [];
  if (contribs.length === 0) return null;
  return (
    <div className="border-t pt-2 space-y-1">
      <p className="text-xs font-medium">
        Top model contributions (
        <TermHelp term="SHAP">SHAP</TermHelp>, toward home win):
      </p>
      <ul className="text-xs space-y-0.5">
        {contribs.map((c) => {
          const sign = c.contribution >= 0 ? "↑" : "↓";
          return (
            <li key={c.feature}>
              {sign}{" "}
              <span className="font-medium">{c.feature}</span> ({signed(c.contribution, 3)})
            </li>
          );
        })}
      </ul>
    </div>
  );
}
