import { Trophy } from "lucide-react";

import { TeamChip } from "@/components/team-chip";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

export type BracketMatch = {
  match_id: number;
  round: "R32" | "R16" | "QF" | "SF" | "Final";
  home_team: string;
  away_team: string;
  winner: string;
  decided_in: "regulation" | "extra_time" | "shootout";
  regulation_score: [number, number];
};

export type BracketResponse = {
  seed: number;
  champion: string;
  matches: BracketMatch[];
};

const ROUND_ORDER: BracketMatch["round"][] = ["R32", "R16", "QF", "SF", "Final"];
const DECIDED_LABEL = {
  regulation: "90'",
  extra_time: "AET",
  shootout: "pens",
} as const;

export function BracketDetail({ data }: { data: BracketResponse }) {
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2 rounded-md border bg-card px-4 py-3">
        <Trophy className="h-5 w-5 text-amber-500" aria-hidden />
        <span className="text-sm">
          <span className="font-medium">Champion (seed {data.seed}):</span>{" "}
          <TeamChip team={data.champion} bold />
        </span>
      </div>
      {ROUND_ORDER.map((round) => {
        const rows = data.matches.filter((m) => m.round === round);
        if (rows.length === 0) return null;
        return (
          <div key={round}>
            <h3 className="text-sm font-medium mb-1">
              {round} — {rows.length} match{rows.length === 1 ? "" : "es"}
            </h3>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>#</TableHead>
                  <TableHead>Home</TableHead>
                  <TableHead>Score</TableHead>
                  <TableHead>Away</TableHead>
                  <TableHead>Winner</TableHead>
                  <TableHead>Decided</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {rows.map((m) => (
                  <TableRow key={m.match_id}>
                    <TableCell className="text-xs tabular-nums">
                      {m.match_id}
                    </TableCell>
                    <TableCell className="text-xs">{m.home_team}</TableCell>
                    <TableCell className="text-xs tabular-nums">
                      {m.regulation_score[0]}-{m.regulation_score[1]}
                    </TableCell>
                    <TableCell className="text-xs">{m.away_team}</TableCell>
                    <TableCell className="text-xs">
                      <TeamChip team={m.winner} size="sm" bold />
                    </TableCell>
                    <TableCell className="text-xs text-muted-foreground">
                      {DECIDED_LABEL[m.decided_in]}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        );
      })}
    </div>
  );
}
