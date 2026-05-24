import { apiGet, ApiError, ApiUnreachable } from "@/lib/api";
import { ApiUnreachableBanner } from "@/components/api-unreachable-banner";
import { ForecastHeader } from "@/components/forecast-header";
import { TeamChip } from "@/components/team-chip";
import {
  GroupCard,
  type GroupStanding,
  type LiveGroupTeam,
} from "@/components/groups/group-card";
import { NSimsSlider } from "@/components/groups/n-sims-slider";

type StandingsResponse = {
  n_sims: number;
  source: string | null;
  run_id: number | null;
  model_version: string | null;
  groups: { group: string; teams: GroupStanding[] }[];
  headline: {
    team: string;
    p_champion: number;
    p_final: number;
    p_sf: number;
    p_qf: number;
  }[];
};

type GroupsLiveResponse = {
  groups: { group: string; teams: LiveGroupTeam[] }[];
};

const DEFAULT_N_SIMS = 2000;

export default async function GroupsPage({
  searchParams,
}: {
  searchParams: Promise<{ n_sims?: string }>;
}) {
  const params = await searchParams;
  const n_sims = clamp(Number(params.n_sims) || DEFAULT_N_SIMS, 200, 10_000);

  let data: StandingsResponse | null = null;
  let live: GroupsLiveResponse | null = null;
  let unreachable = false;
  try {
    [data, live] = await Promise.all([
      apiGet<StandingsResponse>(
        "/api/v1/tournament/standings",
        { n_sims, seed: 42 },
        { revalidate: 300 },
      ),
      apiGet<GroupsLiveResponse>("/api/v1/tournament/groups-live", undefined, {
        revalidate: 60,
      }).catch(() => null),
    ]);
  } catch (err) {
    if (err instanceof ApiUnreachable) {
      unreachable = true;
    } else if (!(err instanceof ApiError)) {
      throw err;
    }
  }

  if (unreachable || !data) {
    return (
      <div className="space-y-6">
        <h1 className="text-2xl font-semibold">Group-stage advancement</h1>
        <ApiUnreachableBanner />
      </div>
    );
  }

  const provenance =
    data.source === "persisted" && data.run_id != null
      ? `Based on ${data.n_sims.toLocaleString()} simulations · persisted run #${data.run_id} (model ${data.model_version ?? "unknown"}). Updates after each completed match.`
      : `Based on ${data.n_sims.toLocaleString()} simulations · in-process run (no persisted Monte Carlo run on file yet).`;

  const liveByGroup = new Map(
    (live?.groups ?? []).map((g) => [g.group, g.teams] as const),
  );

  return (
    <div className="space-y-6">
      <header className="space-y-2">
        <h1 className="text-2xl font-semibold tracking-tight">
          Group-stage advancement probabilities
        </h1>
        <ForecastHeader />
      </header>

      <p className="text-xs text-muted-foreground">
        Each row shows where the model thinks a team will finish. Top 2 + 8 best
        3rd-placed teams advance to the Round of 32. Bars stack: 1st (dark blue),
        2nd (sky), 3rd→R32 (amber), 3rd-out (light), 4th (grey).
      </p>

      <div className="rounded-lg border bg-card p-4">
        <NSimsSlider initial={n_sims} />
        <p className="text-xs text-muted-foreground mt-2 italic">{provenance}</p>
      </div>

      <section className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
        {data.groups.map((block) => (
          <GroupCard
            key={block.group}
            letter={block.group}
            teams={block.teams}
            live={liveByGroup.get(block.group) ?? null}
          />
        ))}
      </section>

      <section className="space-y-3">
        <h2 className="text-lg font-semibold">
          Headline: top 10 championship probabilities
        </h2>
        <div className="grid grid-cols-[3fr_1fr_1fr_1fr_1fr] items-center gap-x-3 text-sm">
          <div className="font-medium">Team</div>
          <div className="text-right font-medium">Champion</div>
          <div className="text-right font-medium">Final</div>
          <div className="text-right font-medium">Semi</div>
          <div className="text-right font-medium">Quarter</div>
          {data.headline.map((h) => (
            <Row key={h.team} {...h} />
          ))}
        </div>
      </section>
    </div>
  );
}

function clamp(n: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, n));
}

function Row({
  team,
  p_champion,
  p_final,
  p_sf,
  p_qf,
}: {
  team: string;
  p_champion: number;
  p_final: number;
  p_sf: number;
  p_qf: number;
}) {
  return (
    <>
      <div className="py-1">
        <TeamChip team={team} bold />
      </div>
      <div className="text-right tabular-nums">{(p_champion * 100).toFixed(1)}%</div>
      <div className="text-right tabular-nums">{(p_final * 100).toFixed(1)}%</div>
      <div className="text-right tabular-nums">{(p_sf * 100).toFixed(1)}%</div>
      <div className="text-right tabular-nums">{(p_qf * 100).toFixed(1)}%</div>
    </>
  );
}
