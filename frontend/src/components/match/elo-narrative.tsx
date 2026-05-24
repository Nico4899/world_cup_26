import { apiGet, ApiError, ApiUnreachable } from "@/lib/api";
import { signed } from "@/lib/format";

type EloHistory = {
  team: string;
  history: { snapshot_date: string; rating: number; global_rank: number | null }[];
};

async function latestElo(team: string): Promise<number | null> {
  try {
    const payload = await apiGet<EloHistory>(
      `/api/v1/teams/${team}/elo-history`,
      undefined,
      { revalidate: 3600 },
    );
    const history = payload.history ?? [];
    if (history.length === 0) return null;
    return history[history.length - 1].rating;
  } catch (err) {
    if (err instanceof ApiUnreachable || err instanceof ApiError) return null;
    throw err;
  }
}

type Props = {
  homeTeam: string;
  awayTeam: string;
  neutral: boolean;
  expectedHomeGoals: number;
  expectedAwayGoals: number;
  topScoreline: { home_goals: number; away_goals: number; probability: number };
};

/**
 * Plain-language "Why this prediction" narrative. Tries to anchor the xG
 * edge sentence to the Elo gap when both teams have a snapshot; otherwise
 * degrades to an xG-only sentence.
 */
export async function EloNarrative({
  homeTeam,
  awayTeam,
  neutral,
  expectedHomeGoals,
  expectedAwayGoals,
  topScoreline,
}: Props) {
  const xgDiff = expectedHomeGoals - expectedAwayGoals;
  const homeAdvNote = neutral
    ? "**at a neutral venue** (home advantage suppressed)"
    : "**at home** in their host country (home advantage applied)";

  let edge: string;
  if (Math.abs(xgDiff) < 0.2) {
    edge = "The model sees this as roughly even on expected goals";
  } else if (xgDiff > 0) {
    edge = `The model gives **${homeTeam} a ${signed(xgDiff)} expected-goal edge**`;
  } else {
    edge = `The model gives **${awayTeam} a ${signed(-xgDiff)} expected-goal edge**`;
  }

  const [eloH, eloA] = await Promise.all([latestElo(homeTeam), latestElo(awayTeam)]);
  let eloSentence: string | null = null;
  if (eloH != null && eloA != null) {
    const diff = eloH - eloA;
    if (Math.abs(diff) < 5) {
      eloSentence = `Elo is essentially level (${eloH.toFixed(0)} vs ${eloA.toFixed(0)}); the model leans on the bivariate Poisson + home-advantage term for the edge.`;
    } else if (diff > 0) {
      eloSentence = `**${homeTeam}'s Elo is ${signed(diff, 0)} above ${awayTeam}'s** (${eloH.toFixed(0)} vs ${eloA.toFixed(0)}); that translates to a **${signed(xgDiff)} expected-goal edge** here.`;
    } else {
      eloSentence = `**${awayTeam}'s Elo is ${signed(-diff, 0)} above ${homeTeam}'s** (${eloA.toFixed(0)} vs ${eloH.toFixed(0)}); that translates to a **${signed(xgDiff)} expected-goal edge** here.`;
    }
  }

  return (
    <ul className="text-sm space-y-1.5 [&_strong]:font-semibold">
      <li>
        Expected goals: <strong>{homeTeam} {expectedHomeGoals.toFixed(2)}</strong>{" "}
        vs <strong>{awayTeam} {expectedAwayGoals.toFixed(2)}</strong>
      </li>
      <li dangerouslySetInnerHTML={{ __html: `${markdownToHtml(edge)}, ${markdownToHtml(homeAdvNote)}.` }} />
      {eloSentence ? (
        <li dangerouslySetInnerHTML={{ __html: markdownToHtml(eloSentence) }} />
      ) : null}
      <li>
        Top scoreline: <strong>{topScoreline.home_goals}-{topScoreline.away_goals}</strong>{" "}
        at {(topScoreline.probability * 100).toFixed(1)}%.
      </li>
      <li>
        Remember: a 60% favourite still loses 40% of the time. These are
        probabilities, not predictions.
      </li>
    </ul>
  );
}

/** Trivial **bold** → <strong> conversion; no HTML injection vector since
 *  the input strings are all controlled by this module. */
function markdownToHtml(s: string): string {
  return s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
}
