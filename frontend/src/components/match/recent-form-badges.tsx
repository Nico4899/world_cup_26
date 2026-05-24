import { apiGet, ApiError, ApiUnreachable } from "@/lib/api";

const COLOR: Record<string, string> = {
  W: "#1f9d55",
  D: "#888888",
  L: "#d62728",
};

type RecentMatch = {
  date: string;
  opponent: string;
  venue: string;
  goals_for: number;
  goals_against: number;
  result: "W" | "D" | "L";
  tournament: string;
};

export async function RecentFormBadges({
  team,
  n = 5,
}: {
  team: string;
  n?: number;
}) {
  let form: RecentMatch[] = [];
  try {
    form = await apiGet<RecentMatch[]>(`/api/v1/teams/${team}/recent`, { n }, {
      revalidate: 300,
    });
  } catch (err) {
    if (err instanceof ApiUnreachable || err instanceof ApiError) {
      form = [];
    } else {
      throw err;
    }
  }
  if (form.length === 0) {
    return (
      <p className="text-xs text-muted-foreground">No recent matches in dataset.</p>
    );
  }
  return (
    <div className="space-y-1">
      <div className="flex flex-wrap gap-1">
        {form.map((m, i) => (
          <span
            key={i}
            title={`${m.date} ${team} ${m.goals_for}-${m.goals_against} ${m.opponent} (${m.venue}, ${m.tournament})`}
            className="inline-flex items-center justify-center rounded px-2 py-0.5 text-xs font-semibold text-white"
            style={{ background: COLOR[m.result] ?? "#666" }}
          >
            {m.result}
          </span>
        ))}
      </div>
      <p className="text-[11px] text-muted-foreground">
        {form
          .map(
            (m) =>
              `${m.result} ${m.goals_for}-${m.goals_against} ${
                m.venue === "away" ? "@" : "vs"
              } ${m.opponent}`,
          )
          .join(" · ")}
      </p>
    </div>
  );
}
