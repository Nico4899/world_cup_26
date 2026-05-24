"use client";

import { Suspense, useMemo, useState } from "react";
import { useRouter, useSearchParams, usePathname } from "next/navigation";
import { useMutation, useQueries, useQuery } from "@tanstack/react-query";
import { Trash2 } from "lucide-react";

import { ApiUnreachableBanner } from "@/components/api-unreachable-banner";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { TeamChip } from "@/components/team-chip";
import {
  BracketDetail,
  type BracketMatch,
  type BracketResponse,
} from "@/components/bracket/bracket-detail";
import { ApiError, ApiUnreachable, apiGet, apiPost } from "@/lib/api";
import { pct } from "@/lib/format";
import { useLockedBracket, type Lock } from "@/hooks/use-locked-bracket";
import type { FixtureSummary } from "@/lib/types";

type Mode = "single" | "scenarios" | "locks";
const MODE_LABEL: Record<Mode, string> = {
  single: "Single seed",
  scenarios: "Scenario comparison",
  locks: "Conditional locks",
};

type Round = "R32" | "R16" | "QF" | "SF" | "Final";
const ROUND_RANGES: { round: Round; ids: number[] }[] = [
  { round: "R32", ids: rangeIds(73, 89) },
  { round: "R16", ids: rangeIds(89, 97) },
  { round: "QF", ids: rangeIds(97, 101) },
  { round: "SF", ids: rangeIds(101, 103) },
  { round: "Final", ids: [104] },
];

function rangeIds(lo: number, hi: number) {
  return Array.from({ length: hi - lo }, (_, i) => lo + i);
}

function clamp(n: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, n));
}

export default function BracketPage() {
  return (
    <Suspense
      fallback={<p className="text-sm text-muted-foreground">Loading bracket page…</p>}
    >
      <BracketPageInner />
    </Suspense>
  );
}

function BracketPageInner() {
  const router = useRouter();
  const pathname = usePathname();
  const params = useSearchParams();

  const mode: Mode =
    params.get("mode") === "scenarios"
      ? "scenarios"
      : params.get("mode") === "locks"
        ? "locks"
        : "single";
  const seed = clamp(Number(params.get("seed")) || 42, 0, 1_000_000);
  const scenarios = clamp(Number(params.get("scenarios")) || 4, 2, 8);
  const baseSeed = clamp(Number(params.get("base_seed")) || 42, 0, 1_000_000);

  function setParam(key: string, value: string | null) {
    const next = new URLSearchParams(params.toString());
    if (value == null) next.delete(key);
    else next.set(key, value);
    router.replace(`${pathname}?${next.toString()}`);
  }

  return (
    <div className="space-y-6">
      <header className="space-y-2">
        <h1 className="text-2xl font-semibold tracking-tight">
          Knockout bracket realisation(s)
        </h1>
        <p className="text-xs text-muted-foreground">
          Each seed gives one Monte Carlo sample of the full knockout bracket.
          Compare scenarios to see how much the predicted champion / finalists
          vary across realisations. Per-team probabilities are on the Groups page.
        </p>
      </header>

      <Tabs
        value={mode}
        onValueChange={(v) => setParam("mode", v === "single" ? null : v)}
      >
        <TabsList>
          {(["single", "scenarios", "locks"] as Mode[]).map((m) => (
            <TabsTrigger key={m} value={m}>
              {MODE_LABEL[m]}
            </TabsTrigger>
          ))}
        </TabsList>

        <TabsContent value="single" className="space-y-4 pt-4">
          <div className="max-w-xs">
            <Label htmlFor="seed-input" className="text-xs">
              Seed
            </Label>
            <Input
              id="seed-input"
              type="number"
              min={0}
              max={1_000_000}
              defaultValue={seed}
              onChange={(e) => setParam("seed", e.target.value || null)}
            />
          </div>
          <SingleSeed seed={seed} />
        </TabsContent>

        <TabsContent value="scenarios" className="space-y-4 pt-4">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 max-w-md">
            <div>
              <Label className="text-xs">Number of scenarios</Label>
              <Slider
                min={2}
                max={8}
                step={1}
                value={[scenarios]}
                onValueCommitted={(v) => {
                  const next = Array.isArray(v) ? v[0] : v;
                  if (typeof next === "number") setParam("scenarios", String(next));
                }}
              />
              <p className="text-xs text-muted-foreground tabular-nums mt-1">
                {scenarios}
              </p>
            </div>
            <div>
              <Label className="text-xs">Base seed (first scenario)</Label>
              <Input
                type="number"
                min={0}
                max={1_000_000}
                defaultValue={baseSeed}
                onChange={(e) => setParam("base_seed", e.target.value || null)}
              />
            </div>
          </div>
          <ScenarioComparison count={scenarios} baseSeed={baseSeed} />
        </TabsContent>

        <TabsContent value="locks" className="space-y-4 pt-4">
          <ConditionalLocks />
        </TabsContent>
      </Tabs>
    </div>
  );
}

function SingleSeed({ seed }: { seed: number }) {
  const { data, error, isPending } = useQuery({
    queryKey: ["bracket", seed],
    queryFn: () =>
      apiGet<BracketResponse>("/api/v1/tournament/bracket", { seed }),
    staleTime: 5 * 60 * 1000,
    retry: false,
  });
  if (error instanceof ApiUnreachable) return <ApiUnreachableBanner />;
  if (error instanceof ApiError)
    return <p className="text-sm text-destructive">HTTP {error.status}</p>;
  if (error) return <p className="text-sm text-destructive">{String(error)}</p>;
  if (isPending || !data)
    return <p className="text-sm text-muted-foreground">Loading…</p>;
  return <BracketDetail data={data} />;
}

function ScenarioComparison({
  count,
  baseSeed,
}: {
  count: number;
  baseSeed: number;
}) {
  const queries = useQueries({
    queries: Array.from({ length: count }, (_, i) => ({
      queryKey: ["bracket", baseSeed + i],
      queryFn: () =>
        apiGet<BracketResponse>("/api/v1/tournament/bracket", { seed: baseSeed + i }),
      staleTime: 5 * 60 * 1000,
    })),
  });

  if (queries.some((q) => q.isPending)) {
    return <p className="text-sm text-muted-foreground">Loading {count} scenarios…</p>;
  }
  const errored = queries.find((q) => q.error);
  if (errored?.error instanceof ApiUnreachable) return <ApiUnreachableBanner />;
  if (errored)
    return <p className="text-sm text-destructive">Failed to load scenarios.</p>;
  const scenarios = queries.map((q) => q.data!) as BracketResponse[];

  const sfCount: Record<string, number> = {};
  const finalsCount: Record<string, number> = {};
  const champsCount: Record<string, number> = {};
  for (const s of scenarios) {
    const sfMatches = s.matches.filter((m: BracketMatch) => m.round === "SF");
    for (const sf of sfMatches) {
      sfCount[sf.home_team] = (sfCount[sf.home_team] ?? 0) + 1;
      sfCount[sf.away_team] = (sfCount[sf.away_team] ?? 0) + 1;
    }
    const finalMatch = s.matches.find((m: BracketMatch) => m.round === "Final");
    if (finalMatch) {
      finalsCount[finalMatch.home_team] = (finalsCount[finalMatch.home_team] ?? 0) + 1;
      finalsCount[finalMatch.away_team] = (finalsCount[finalMatch.away_team] ?? 0) + 1;
    }
    champsCount[s.champion] = (champsCount[s.champion] ?? 0) + 1;
  }

  const teams = new Set<string>([
    ...Object.keys(sfCount),
    ...Object.keys(finalsCount),
    ...Object.keys(champsCount),
  ]);
  const rows = [...teams]
    .map((team) => ({
      team,
      sfs: sfCount[team] ?? 0,
      finals: finalsCount[team] ?? 0,
      champion: champsCount[team] ?? 0,
    }))
    .sort(
      (a, b) =>
        b.champion * 100 + b.finals * 10 + b.sfs - (a.champion * 100 + a.finals * 10 + a.sfs),
    );

  return (
    <div className="space-y-4">
      <h2 className="text-lg font-semibold">Across {count} scenarios</h2>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Team</TableHead>
            <TableHead className="text-right">SFs</TableHead>
            <TableHead className="text-right">Finals</TableHead>
            <TableHead className="text-right">Champion</TableHead>
            <TableHead className="text-right">Champion %</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((r) => (
            <TableRow key={r.team}>
              <TableCell>
                <TeamChip team={r.team} bold />
              </TableCell>
              <TableCell className="text-right tabular-nums">{r.sfs}</TableCell>
              <TableCell className="text-right tabular-nums">{r.finals}</TableCell>
              <TableCell className="text-right tabular-nums">{r.champion}</TableCell>
              <TableCell className="text-right tabular-nums">
                {pct(r.champion / count, 0)}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
        {scenarios.map((s) => (
          <Card key={s.seed}>
            <CardContent className="py-3">
              <p className="text-xs text-muted-foreground">Seed {s.seed}</p>
              <TeamChip team={s.champion} bold />
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}

type ConditionalResponse = {
  n_sims: number;
  seed: number;
  locks: Lock[];
  headline: {
    team: string;
    p_champion: number;
    p_final: number;
    p_sf: number;
    p_qf: number;
  }[];
};

function ConditionalLocks() {
  const { locks, add, clear } = useLockedBracket();
  const [nSims, setNSims] = useState(5000);
  const [seed, setSeed] = useState(42);
  const [round, setRound] = useState<Round>("R32");
  const [matchId, setMatchId] = useState<number>(73);
  const [winner, setWinner] = useState<string>("");

  const matches = useQueries({
    queries: [
      {
        queryKey: ["matches-list"],
        queryFn: () => apiGet<FixtureSummary[]>("/api/v1/matches"),
        staleTime: 5 * 60 * 1000,
      },
    ],
  });
  const teams = useMemo(() => {
    const list = matches[0].data ?? [];
    const set = new Set<string>();
    for (const m of list) {
      set.add(m.home_team);
      set.add(m.away_team);
    }
    return [...set].sort();
  }, [matches]);

  const mutation = useMutation({
    mutationFn: (payload: { locks: Lock[]; n_sims: number; seed: number }) =>
      apiPost<ConditionalResponse>(
        "/api/v1/tournament/bracket/conditional",
        payload,
      ),
  });

  const matchIds = useMemo(
    () => ROUND_RANGES.find((r) => r.round === round)?.ids ?? [],
    [round],
  );

  function addLock(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (!winner) return;
    add({ match_id: matchId, winner });
  }

  return (
    <div className="space-y-6">
      <p className="text-xs text-muted-foreground">
        Lock one or more knockout match winners, then run a 5,000-sim conditional
        Monte Carlo. Locks that aren&apos;t reachable in a given sim are skipped
        — the rest of the bracket plays out normally.
      </p>

      <form onSubmit={addLock} className="grid grid-cols-1 sm:grid-cols-4 gap-3 items-end">
        <div className="space-y-1">
          <Label className="text-xs">Round</Label>
          <Select
            value={round}
            onValueChange={(v) => {
              const next = v as Round;
              setRound(next);
              const ids = ROUND_RANGES.find((r) => r.round === next)?.ids ?? [];
              setMatchId(ids[0] ?? 73);
            }}
          >
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {ROUND_RANGES.map((r) => (
                <SelectItem key={r.round} value={r.round}>
                  {r.round}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="space-y-1">
          <Label className="text-xs">Match ID</Label>
          <Select value={String(matchId)} onValueChange={(v) => setMatchId(Number(v))}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {matchIds.map((id) => (
                <SelectItem key={id} value={String(id)}>
                  #{id}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="space-y-1">
          <Label className="text-xs">Winner</Label>
          <Select value={winner} onValueChange={(v) => setWinner(v ?? "")}>
            <SelectTrigger>
              <SelectValue placeholder="Pick team" />
            </SelectTrigger>
            <SelectContent>
              {teams.map((t) => (
                <SelectItem key={t} value={t}>
                  {t}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <Button type="submit" disabled={!winner}>
          Add lock
        </Button>
      </form>

      {locks.length > 0 ? (
        <Card>
          <CardHeader className="pb-2 flex flex-row items-center justify-between space-y-0">
            <CardTitle className="text-base">Active locks</CardTitle>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => clear()}
              data-icon="inline-start"
            >
              <Trash2 className="h-3.5 w-3.5 mr-1" aria-hidden />
              Clear all
            </Button>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Match #</TableHead>
                  <TableHead>Winner</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {locks.map((l) => (
                  <TableRow key={l.match_id}>
                    <TableCell className="tabular-nums">{l.match_id}</TableCell>
                    <TableCell>
                      <TeamChip team={l.winner} bold />
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      ) : (
        <p className="text-xs text-muted-foreground">
          No locks yet. Use the form above to pin a knockout winner.
        </p>
      )}

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 max-w-md">
        <div>
          <Label className="text-xs">Monte Carlo simulations</Label>
          <Slider
            min={500}
            max={10_000}
            step={500}
            value={[nSims]}
            onValueChange={(v) => {
              const next = Array.isArray(v) ? v[0] : v;
              if (typeof next === "number") setNSims(next);
            }}
          />
          <p className="text-xs text-muted-foreground tabular-nums mt-1">
            {nSims.toLocaleString()}
          </p>
        </div>
        <div>
          <Label className="text-xs">Seed</Label>
          <Input
            type="number"
            min={0}
            max={1_000_000}
            value={seed}
            onChange={(e) => setSeed(clamp(Number(e.target.value) || 42, 0, 1_000_000))}
          />
        </div>
      </div>

      <Button
        onClick={() => mutation.mutate({ locks, n_sims: nSims, seed })}
        disabled={mutation.isPending}
      >
        {mutation.isPending ? "Running…" : "Run conditional MC"}
      </Button>

      {mutation.error instanceof ApiUnreachable ? <ApiUnreachableBanner /> : null}
      {mutation.data ? (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">
              Headline: top 10 championship probabilities
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-xs text-muted-foreground italic mb-2">
              Ran {mutation.data.n_sims.toLocaleString()} sims with{" "}
              {mutation.data.locks.length} active lock(s) (seed {mutation.data.seed}).
            </p>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Team</TableHead>
                  <TableHead className="text-right">Champion</TableHead>
                  <TableHead className="text-right">Final</TableHead>
                  <TableHead className="text-right">Semi</TableHead>
                  <TableHead className="text-right">Quarter</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {mutation.data.headline.map((h) => (
                  <TableRow key={h.team}>
                    <TableCell>
                      <TeamChip team={h.team} bold />
                    </TableCell>
                    <TableCell className="text-right tabular-nums">
                      {pct(h.p_champion)}
                    </TableCell>
                    <TableCell className="text-right tabular-nums">
                      {pct(h.p_final)}
                    </TableCell>
                    <TableCell className="text-right tabular-nums">
                      {pct(h.p_sf)}
                    </TableCell>
                    <TableCell className="text-right tabular-nums">
                      {pct(h.p_qf)}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      ) : null}
    </div>
  );
}
