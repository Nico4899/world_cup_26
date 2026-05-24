"use client";

import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Sparkles } from "lucide-react";

import { ApiError, apiGet } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import {
  Sheet,
  SheetContent,
  SheetTrigger,
  SheetTitle,
  SheetDescription,
  SheetHeader,
} from "@/components/ui/sheet";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { buttonVariants } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { pct } from "@/lib/format";
import type { PredictionResponse } from "@/lib/types";

type Props = {
  homeTeam: string;
  awayTeam: string;
  neutral: boolean;
};

/**
 * Phase 5 Poisson × XGB blend overlay.
 *
 * Sheet-based UI with a checkbox + weight slider so it doesn't push the
 * headline tiles off the Match Detail viewport. The blend is opt-in
 * because WC 2018 / 2022 hindcasts show it regressing log-loss — kept as
 * a research artefact, not the default outcome.
 */
export function BlendOverlay({ homeTeam, awayTeam, neutral }: Props) {
  const [showBlend, setShowBlend] = useState(false);
  const [poissonWeight, setPoissonWeight] = useState(0.5);

  const { data, error, isFetching } = useQuery({
    queryKey: ["predictions-blend", homeTeam, awayTeam, neutral, poissonWeight],
    queryFn: () =>
      apiGet<PredictionResponse>(`/api/v1/predictions/${homeTeam}/${awayTeam}`, {
        neutral: String(neutral),
        blend: "true",
        blend_weight: poissonWeight,
      }),
    enabled: showBlend,
    retry: false,
    staleTime: 5 * 60 * 1000,
  });

  return (
    <Sheet>
      <SheetTrigger
        className={cn(buttonVariants({ variant: "outline", size: "sm" }))}
      >
        <Sparkles className="h-3.5 w-3.5 mr-1.5" aria-hidden />
        XGB blend
      </SheetTrigger>
      <SheetContent side="right" className="w-95 sm:max-w-md">
        <SheetHeader>
          <SheetTitle>XGB blend (Phase 5)</SheetTitle>
          <SheetDescription>
            Overlay the Phase 5 XGBoost classifier on the Poisson outcome.
            WC 2018/2022 hindcasts show this regressing log-loss, so it&apos;s
            a research artefact rather than the default — opt in to compare.
          </SheetDescription>
        </SheetHeader>

        <div className="space-y-4 mt-4 px-4">
          <Label className="flex items-center justify-between gap-2">
            <span className="text-sm">Show Poisson × XGB blend</span>
            <Switch checked={showBlend} onCheckedChange={setShowBlend} />
          </Label>

          <div className="space-y-1">
            <Label className="text-xs">
              Poisson weight ({poissonWeight.toFixed(2)})
            </Label>
            <Slider
              min={0}
              max={1}
              step={0.05}
              value={[poissonWeight]}
              onValueChange={(v) => {
                const next = Array.isArray(v) ? v[0] : v;
                if (typeof next === "number") setPoissonWeight(next);
              }}
              disabled={!showBlend}
            />
            <p className="text-[11px] text-muted-foreground">
              Geometric-mean mix; XGB gets {(1 - poissonWeight).toFixed(2)}.
            </p>
          </div>

          {showBlend ? <BlendBody isFetching={isFetching} data={data} error={error} homeTeam={homeTeam} awayTeam={awayTeam} /> : null}
        </div>
      </SheetContent>
    </Sheet>
  );
}

function BlendBody({
  isFetching,
  data,
  error,
  homeTeam,
  awayTeam,
}: {
  isFetching: boolean;
  data: PredictionResponse | undefined;
  error: unknown;
  homeTeam: string;
  awayTeam: string;
}) {
  if (error) {
    const status = error instanceof ApiError ? error.status : null;
    return (
      <p className="text-xs text-muted-foreground">
        {status === 503
          ? "XGB artefact not loaded server-side. Train it with `uv run python scripts/refit_xgb.py` and restart the API."
          : "Blend overlay unavailable."}
      </p>
    );
  }
  if (isFetching || !data) {
    return <p className="text-xs text-muted-foreground">Computing blend…</p>;
  }
  const payload = data.blend;
  if (!payload) {
    return (
      <p className="text-xs text-muted-foreground">
        Blend payload empty — XGB likely not loaded.
      </p>
    );
  }
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm">Blended outcome</CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        <Row label={homeTeam} {...payload} pick="home_win" />
        <Row label="Draw" {...payload} pick="draw" />
        <Row label={awayTeam} {...payload} pick="away_win" />
        <p className="text-[11px] text-muted-foreground pt-1 border-t">
          Poisson weight {payload.weight.toFixed(2)}, XGB weight{" "}
          {(1 - payload.weight).toFixed(2)}.
        </p>
      </CardContent>
    </Card>
  );
}

function Row({
  label,
  poisson,
  xgb,
  blended,
  pick,
}: {
  label: string;
  poisson: { home_win: number; draw: number; away_win: number };
  xgb: { home_win: number; draw: number; away_win: number };
  blended: { home_win: number; draw: number; away_win: number };
  pick: "home_win" | "draw" | "away_win";
}) {
  return (
    <div className="grid grid-cols-[1fr_auto_auto_auto] gap-2 text-xs items-baseline">
      <span className="font-medium truncate">{label}</span>
      <span className="tabular-nums text-muted-foreground">P {pct(poisson[pick], 0)}</span>
      <span className="tabular-nums text-muted-foreground">X {pct(xgb[pick], 0)}</span>
      <span className="tabular-nums font-semibold">{pct(blended[pick])}</span>
    </div>
  );
}
