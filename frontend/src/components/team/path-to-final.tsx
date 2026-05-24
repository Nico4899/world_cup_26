import { Card, CardContent } from "@/components/ui/card";
import { pct } from "@/lib/format";

type PathOpponent = { team: string; p_conditional: number };
type PathRound = {
  round: "r32" | "r16" | "qf" | "sf" | "final";
  p_reached: number;
  most_likely_opponent: PathOpponent | null;
  top_opponents: PathOpponent[];
};

const LABEL: Record<PathRound["round"], string> = {
  r32: "R32",
  r16: "R16",
  qf: "QF",
  sf: "SF",
  final: "Final",
};

export function PathToFinal({
  rounds,
  nSims,
}: {
  rounds: PathRound[];
  nSims: number;
}) {
  if (!rounds.length || rounds.every((r) => (r.p_reached ?? 0) === 0)) {
    return (
      <p className="text-xs text-muted-foreground italic">
        No knockout reach in this MC sample. Most-likely-opponent estimates appear
        once the team reaches a round in at least one simulation.
      </p>
    );
  }
  return (
    <div className="space-y-2">
      <div className="grid grid-cols-2 sm:grid-cols-5 gap-2">
        {rounds.map((r) => (
          <Card key={r.round}>
            <CardContent className="py-3 space-y-1">
              <p className="text-xs font-semibold">{LABEL[r.round]}</p>
              <p className="text-lg font-semibold tabular-nums">{pct(r.p_reached)}</p>
              {r.most_likely_opponent ? (
                <p className="text-[11px] text-muted-foreground">
                  Most likely opp:{" "}
                  <strong>{r.most_likely_opponent.team}</strong>{" "}
                  ({pct(r.most_likely_opponent.p_conditional, 0)})
                </p>
              ) : (
                <p className="text-[11px] text-muted-foreground italic">
                  no opponent observed
                </p>
              )}
            </CardContent>
          </Card>
        ))}
      </div>
      <p className="text-xs text-muted-foreground italic">
        From a {nSims.toLocaleString()}-sim Monte Carlo pass.
      </p>
    </div>
  );
}
