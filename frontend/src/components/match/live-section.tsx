"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

/**
 * Phase D will replace this with a live-win-prob chart driven by an
 * EventSource subscription. For now the page renders this only when the API
 * already reports a non-pre-match snapshot for the fixture.
 */
export function LiveSection({ matchId }: { matchId: number }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Live (Phase D)</CardTitle>
      </CardHeader>
      <CardContent>
        <p className="text-xs text-muted-foreground">
          Live win-prob streaming for match {matchId} lands in Phase D
          (useLiveWinProb hook + Visx scatter with goal/red annotations).
        </p>
      </CardContent>
    </Card>
  );
}
