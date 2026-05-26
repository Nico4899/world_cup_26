import { apiGet, ApiError, ApiUnreachable } from "@/lib/api";
import { TeamChip } from "@/components/team-chip";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

type StandingsTeam = {
  team: string;
  p_first: number;
  p_second: number;
  p_third_advance: number;
  p_third_out: number;
  p_fourth: number;
  p_eliminated: number;
};

type StandingsResponse = {
  n_sims: number;
  groups: { group: string; teams: StandingsTeam[] }[];
  source?: string | null;
  run_id?: number | null;
  model_version?: string | null;
};

/** Compact 4-column grid showing per-team advance% across all 12 groups. */
export async function GroupStrip() {
  let data: StandingsResponse | null = null;
  try {
    data = await apiGet<StandingsResponse>(
      "/api/v1/tournament/standings",
      { n_sims: 2000, seed: 42 },
      { revalidate: 600 },
    );
  } catch (err) {
    if (err instanceof ApiUnreachable || err instanceof ApiError) {
      return null;
    }
    throw err;
  }
  if (!data) return null;

  return (
    <section className="space-y-2">
      <h2 className="text-lg font-semibold">
        Group-stage advancement (across all 12 groups)
      </h2>
      <p className="text-xs text-muted-foreground">
        Stacked-bar advancement probabilities from the Monte Carlo simulator.{" "}
        Each row is one group; the percentage is P(1st + 2nd + 3rd→R32).{" "}
        For per-team detail see the <strong>Groups</strong> page.
      </p>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
        {data.groups.map((block) => (
          <Card key={block.group}>
            <CardHeader className="pb-1">
              <CardTitle className="text-sm">Group {block.group}</CardTitle>
            </CardHeader>
            <CardContent className="space-y-1 pt-1">
              {block.teams.map((t) => {
                const adv = t.p_first + t.p_second + t.p_third_advance;
                const filled = Math.round(adv * 12);
                const bar = "█".repeat(filled) + "░".repeat(Math.max(0, 12 - filled));
                return (
                  <div
                    key={t.team}
                    className="flex items-center gap-2 text-xs whitespace-nowrap"
                  >
                    <code className="text-[10px] font-mono">{bar}</code>
                    <TeamChip team={t.team} size="sm" asLink />
                    <span className="text-muted-foreground">
                      · {Math.round(adv * 100)}% adv
                    </span>
                  </div>
                );
              })}
            </CardContent>
          </Card>
        ))}
      </div>
    </section>
  );
}
