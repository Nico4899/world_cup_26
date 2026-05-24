"use client";

import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";

import { ApiUnreachable, ApiError, apiGet } from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import {
  LiveWinProbChart,
} from "@/components/match/live-win-prob-chart";
import { useLiveWinProb, type LiveSnapshot } from "@/hooks/use-live-win-prob";
import { pct } from "@/lib/format";

type LiveEventTrace = {
  seq: number;
  minute: number;
  period: number;
  event_type: string;
  team: string | null;
  home_score_after: number;
  away_score_after: number;
  home_red_cards_after: number;
  away_red_cards_after: number;
  win_prob: { home_win: number; draw: number; away_win: number };
};

type LiveHistory = {
  snapshot: LiveSnapshot;
  events: LiveEventTrace[];
};

function traceToSnapshot(
  t: LiveEventTrace,
  home: string,
  away: string,
  source: string,
): LiveSnapshot {
  return {
    match_id: -1, // not used by the chart
    home_team: home,
    away_team: away,
    minute: t.minute,
    period: t.period,
    home_score: t.home_score_after,
    away_score: t.away_score_after,
    home_red_cards: t.home_red_cards_after,
    away_red_cards: t.away_red_cards_after,
    last_event_type: t.event_type,
    win_prob: t.win_prob,
    win_prob_source: source,
  };
}

export function LiveSection({ matchId }: { matchId: number }) {
  const { data, isLoading, error } = useQuery({
    queryKey: ["live-history", matchId],
    queryFn: () => apiGet<LiveHistory>(`/api/v1/live/${matchId}/history`),
    staleTime: 0,
    retry: false,
  });

  // Only subscribe to SSE while the match is actually in-play.
  const enabled = data?.snapshot?.win_prob_source === "live_win_prob";
  const live = useLiveWinProb(matchId, { enabled });

  const baseEvents = useMemo<LiveSnapshot[]>(() => {
    if (!data) return [];
    return data.events.map((e) =>
      traceToSnapshot(
        e,
        data.snapshot.home_team,
        data.snapshot.away_team,
        data.snapshot.win_prob_source,
      ),
    );
  }, [data]);
  const merged = useMemo<LiveSnapshot[]>(
    () => [...baseEvents, ...live.events],
    [baseEvents, live.events],
  );

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Live</CardTitle>
        </CardHeader>
        <CardContent>
          <Skeleton className="h-32" />
        </CardContent>
      </Card>
    );
  }
  if (error instanceof ApiError && error.status === 404) {
    // Older API without the /live route — silently skip.
    return null;
  }
  if (error instanceof ApiUnreachable || !data) return null;

  const src = data.snapshot.win_prob_source;
  if (src !== "live_win_prob" && src !== "final") return null;

  const snapshot = live.snapshot ?? data.snapshot;
  const isLive = snapshot.win_prob_source === "live_win_prob";

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0">
        <CardTitle className="text-base">
          {isLive ? "Live" : "Final"}
        </CardTitle>
        <Badge
          variant={isLive ? "destructive" : "default"}
          className={isLive ? "bg-red-500 text-white" : ""}
        >
          {isLive ? "🔴 LIVE" : "✅ FULL TIME"}
        </Badge>
      </CardHeader>
      <CardContent className="space-y-4">
        <div>
          <h3 className="text-lg font-semibold">
            {snapshot.home_team} {snapshot.home_score} - {snapshot.away_score}{" "}
            {snapshot.away_team}
          </h3>
          <p className="text-xs text-muted-foreground">
            min {snapshot.minute} · last: {snapshot.last_event_type}
            {snapshot.home_red_cards || snapshot.away_red_cards
              ? ` · 🟥 ${snapshot.home_red_cards}-${snapshot.away_red_cards}`
              : ""}
          </p>
        </div>
        <div className="grid grid-cols-3 gap-3">
          <Cell label={`${snapshot.home_team} (live)`} value={snapshot.win_prob.home_win} />
          <Cell label="Draw (live)" value={snapshot.win_prob.draw} />
          <Cell label={`${snapshot.away_team} (live)`} value={snapshot.win_prob.away_win} />
        </div>
        {merged.length > 0 ? (
          <LiveWinProbChart
            events={merged}
            homeTeam={snapshot.home_team}
            awayTeam={snapshot.away_team}
          />
        ) : null}
        {isLive && live.status === "error" ? (
          <p className="text-xs text-destructive">
            Live stream disconnected — retrying.
          </p>
        ) : null}
      </CardContent>
    </Card>
  );
}

function Cell({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-md border p-3">
      <p className="text-xs uppercase tracking-wide text-muted-foreground">
        {label}
      </p>
      <p className="text-xl font-semibold tabular-nums">{pct(value)}</p>
    </div>
  );
}
