"use client";

import { useQuery } from "@tanstack/react-query";

import { ApiError, apiGet } from "@/lib/api";
import { MetricCard } from "@/components/metric-card";
import { pct, signed } from "@/lib/format";

type Contribution = { feature: string; value: number | null; contribution: number };
type Explanation = { contributions: Contribution[] };

type Outcome = "home_win" | "draw" | "away_win";

type Props = {
  matchId: number;
  homeTeam: string;
  awayTeam: string;
  probs: { home_win: number; draw: number; away_win: number };
};

/**
 * Three clickable probability tiles for the Match Detail header. Each tile
 * opens a popover with SHAP top-3 contributions for that class, fetched
 * lazily on open via TanStack Query.
 */
export function OutcomeTiles({ matchId, homeTeam, awayTeam, probs }: Props) {
  return (
    <div className="grid grid-cols-3 gap-3">
      <ShapTile
        label={homeTeam}
        value={pct(probs.home_win)}
        matchId={matchId}
        outcome="home_win"
      />
      <ShapTile
        label="Draw"
        value={pct(probs.draw)}
        matchId={matchId}
        outcome="draw"
      />
      <ShapTile
        label={awayTeam}
        value={pct(probs.away_win)}
        matchId={matchId}
        outcome="away_win"
      />
    </div>
  );
}

function ShapTile({
  label,
  value,
  matchId,
  outcome,
}: {
  label: string;
  value: string;
  matchId: number;
  outcome: Outcome;
}) {
  return (
    <MetricCard
      label={label}
      value={value}
      popoverTitle={`Why ${label} = ${value}?`}
      popover={<ShapPopoverBody matchId={matchId} outcome={outcome} />}
    />
  );
}

function ShapPopoverBody({ matchId, outcome }: { matchId: number; outcome: Outcome }) {
  const { data, error, isLoading } = useQuery({
    queryKey: ["explain", matchId, outcome, 3],
    queryFn: () =>
      apiGet<Explanation>(`/api/v1/explain/${matchId}`, {
        class_name: outcome,
        top_n: 3,
      }),
    staleTime: 5 * 60 * 1000,
    retry: false,
  });

  if (isLoading) {
    return <p className="text-xs">Loading SHAP top-3…</p>;
  }
  if (error) {
    const status = error instanceof ApiError ? error.status : null;
    if (status === 503) {
      return (
        <p className="text-xs">
          SHAP explanations require the optional XGB classifier. Train it with{" "}
          <code>uv run python scripts/refit_xgb.py</code> and restart the API.
        </p>
      );
    }
    return <p className="text-xs">Explanation endpoint unavailable.</p>;
  }
  const contribs = data?.contributions ?? [];
  if (contribs.length === 0) {
    return <p className="text-xs">No contributions returned.</p>;
  }
  return (
    <ul className="space-y-1">
      {contribs.map((c) => {
        const sign = c.contribution >= 0 ? "↑" : "↓";
        return (
          <li key={c.feature} className="flex items-start gap-2 text-xs">
            <span aria-hidden>{sign}</span>
            <span>
              <strong>{c.feature}</strong> (value{" "}
              {c.value == null ? "—" : signed(c.value, 3)}, contribution{" "}
              {signed(c.contribution, 3)})
            </span>
          </li>
        );
      })}
    </ul>
  );
}
