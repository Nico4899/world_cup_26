import { ClickableProbabilityBar } from "@/components/clickable-probability-bar";
import type { ProbabilitySegment } from "@/components/probability-bar";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export type GroupStanding = {
  team: string;
  p_first: number;
  p_second: number;
  p_third_advance: number;
  p_third_out: number;
  p_fourth: number;
  p_eliminated: number;
};

export type LiveGroupTeam = {
  team: string;
  played: number;
  wins: number;
  draws: number;
  losses: number;
  points: number;
  goals_for: number;
  goals_against: number;
  goal_difference: number;
};

const SEGMENT_COLORS = {
  first: "#1f4e79",
  second: "#5b9bd5",
  thirdAdv: "#ed7d31",
  thirdOut: "#d9a679",
  fourth: "#a6a6a6",
};

export function GroupCard({
  letter,
  teams,
  live,
  caption,
}: {
  letter: string;
  teams: GroupStanding[];
  live: LiveGroupTeam[] | null;
  /** Provenance line shown inside each per-team popover. */
  caption?: string;
}) {
  const hasLive = live && live.some((r) => r.played > 0);
  // Order teams by MC win prob descending for display.
  const ordered = [...teams].sort(
    (a, b) =>
      b.p_first +
      b.p_second +
      b.p_third_advance -
      (a.p_first + a.p_second + a.p_third_advance),
  );

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-base">Group {letter}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {hasLive ? (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Team</TableHead>
                <TableHead className="text-right">P</TableHead>
                <TableHead className="text-right">Pts</TableHead>
                <TableHead className="text-right">GD</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {live!
                .slice()
                .sort(
                  (a, b) =>
                    b.points - a.points ||
                    b.goal_difference - a.goal_difference ||
                    b.goals_for - a.goals_for,
                )
                .map((r) => (
                  <TableRow key={r.team}>
                    <TableCell className="text-xs">{r.team}</TableCell>
                    <TableCell className="text-xs text-right tabular-nums">
                      {r.played}
                    </TableCell>
                    <TableCell className="text-xs text-right tabular-nums font-medium">
                      {r.points}
                    </TableCell>
                    <TableCell className="text-xs text-right tabular-nums">
                      {r.goal_difference >= 0
                        ? `+${r.goal_difference}`
                        : r.goal_difference}
                    </TableCell>
                  </TableRow>
                ))}
            </TableBody>
          </Table>
        ) : null}
        <div className="space-y-1.5">
          {ordered.map((t) => {
            const segments: ProbabilitySegment[] = [
              { label: "1st", value: t.p_first, color: SEGMENT_COLORS.first },
              { label: "2nd", value: t.p_second, color: SEGMENT_COLORS.second },
              {
                label: "3rd → R32",
                value: t.p_third_advance,
                color: SEGMENT_COLORS.thirdAdv,
              },
              {
                label: "3rd-out",
                value: t.p_third_out,
                color: SEGMENT_COLORS.thirdOut,
              },
              { label: "4th", value: t.p_fourth, color: SEGMENT_COLORS.fourth },
            ];
            return (
              <div key={t.team} className="space-y-0.5">
                <p className="text-xs font-medium leading-tight">{t.team}</p>
                <ClickableProbabilityBar
                  title={`${t.team} — Group ${letter}`}
                  segments={segments}
                  caption={caption}
                  height={20}
                />
              </div>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
}
