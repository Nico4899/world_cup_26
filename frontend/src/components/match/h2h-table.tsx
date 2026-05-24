import { apiGet, ApiError, ApiUnreachable } from "@/lib/api";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

type H2HMatch = {
  date: string;
  home_team: string;
  away_team: string;
  home_score: number;
  away_score: number;
  tournament: string;
  neutral: boolean;
};

export async function H2HTable({
  homeTeam,
  awayTeam,
  n = 10,
}: {
  homeTeam: string;
  awayTeam: string;
  n?: number;
}) {
  let rows: H2HMatch[] = [];
  try {
    rows = await apiGet<H2HMatch[]>(`/api/v1/h2h/${homeTeam}/${awayTeam}`, { n }, {
      revalidate: 600,
    });
  } catch (err) {
    if (err instanceof ApiUnreachable || err instanceof ApiError) {
      rows = [];
    } else {
      throw err;
    }
  }
  if (rows.length === 0) {
    return (
      <p className="text-xs text-muted-foreground">
        {homeTeam} and {awayTeam} have never met in the dataset (1872 → present).
      </p>
    );
  }
  const winsHome = rows.filter(
    (m) =>
      (m.home_team === homeTeam && m.home_score > m.away_score) ||
      (m.away_team === homeTeam && m.away_score > m.home_score),
  ).length;
  const draws = rows.filter((m) => m.home_score === m.away_score).length;
  const winsAway = rows.length - winsHome - draws;
  return (
    <div className="space-y-2">
      <p className="text-xs text-muted-foreground">
        Last {rows.length} meetings — <strong>{homeTeam}</strong> {winsHome} W ·{" "}
        <strong>draw</strong> {draws} · <strong>{awayTeam}</strong> {winsAway} W
      </p>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Date</TableHead>
            <TableHead>Home</TableHead>
            <TableHead>Score</TableHead>
            <TableHead>Away</TableHead>
            <TableHead>Tournament</TableHead>
            <TableHead>Neutral</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((m, i) => (
            <TableRow key={i}>
              <TableCell>{m.date}</TableCell>
              <TableCell>{m.home_team}</TableCell>
              <TableCell className="tabular-nums">
                {m.home_score} - {m.away_score}
              </TableCell>
              <TableCell>{m.away_team}</TableCell>
              <TableCell>{m.tournament}</TableCell>
              <TableCell>{m.neutral ? "✓" : ""}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
