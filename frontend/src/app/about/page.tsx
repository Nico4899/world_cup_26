import { apiGet, ApiError, ApiUnreachable } from "@/lib/api";
import { Card, CardContent } from "@/components/ui/card";
import type { HealthResponse } from "@/lib/types";

export const metadata = {
  title: "About / Methodology — WC 2026 Predictions",
};

/**
 * Server-rendered About page. Surfaces the group-letter assignment source
 * (official FIFA draw vs date-derived clique ordering) from /health, then
 * renders an inline summary of the methodology.
 *
 * Phase E will replace the inline summary with an MDX import of
 * `docs/methodology.md` so this stays a single-source-of-truth doc.
 */
export default async function AboutPage() {
  let groupSource = "derived";
  try {
    const health = await apiGet<HealthResponse>("/health", undefined, {
      noStore: true,
    });
    groupSource = health.group_assignment_source ?? "derived";
  } catch (err) {
    if (!(err instanceof ApiUnreachable) && !(err instanceof ApiError)) throw err;
  }

  return (
    <div className="space-y-6 max-w-3xl">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight">
          About / Methodology
        </h1>
      </header>

      <Card>
        <CardContent className="py-3">
          {groupSource === "derived" ? (
            <p className="text-sm">
              <strong>Group letters A–L</strong> are currently <em>derived</em>{" "}
              from the order of fixture dates (Group A = the earliest opener,
              and so on). They may not match FIFA&apos;s official draw letters
              until <code>data/wc2026_group_assignment.json</code> is populated.
            </p>
          ) : (
            <p className="text-sm">
              <strong>Group letters A–L</strong> sourced from{" "}
              <em>{groupSource.replace(/^official:/, "")}</em>.
            </p>
          )}
        </CardContent>
      </Card>

      <section className="prose prose-sm dark:prose-invert max-w-none">
        <h2>0. Lineage</h2>
        <p>This work stands on four public lines of football-forecasting research:</p>
        <ul>
          <li>
            <strong>Elo ratings</strong> as the team-strength backbone — see
            Lasek, Szlávik &amp; Bhulai (2013), <em>The predictive power of
              ranking systems in association football</em>. The eloratings.net
            implementation is our prior + shootout feature.
          </li>
          <li>
            <strong>Bivariate Poisson with low-score correction</strong> — Dixon
            &amp; Coles (1997), <em>Modelling association football scores and
              inefficiencies in the football betting market</em>.
          </li>
          <li>
            <strong>Time-decay weighted MLE for international football</strong>{" "}
            — Groll, Ley, Schauberger &amp; Van Eetvelde (2019), <em>A hybrid
              random forest to predict soccer matches in international
              tournaments</em>.
          </li>
          <li>
            <strong>Public-facing calibration as a product feature</strong> —
            FiveThirtyEight&apos;s Soccer Power Index demonstrated that
            publishing per-team probability tables and transparent backtests
            builds trust without compromising rigor. The Track Record page
            mirrors that practice (live Brier / log-loss / RPS alongside the
            WC 2018 + WC 2022 hindcasts).
          </li>
        </ul>

        <h2>1. Match model</h2>
        <p>
          Each international fixture is modelled as an independent bivariate
          Poisson with the Dixon-Coles low-score correction. Each team has a
          latent attack and defence strength; expected goals are
        </p>
        <pre>
          {`λ_home = exp(attack[home] + defence[away] + home_adv × (1 − neutral))
λ_away = exp(attack[away] + defence[home])`}
        </pre>
        <p>
          Score probabilities factor as <code>Poisson(λ_home) · Poisson(λ_away)</code>{" "}
          with the multiplicative τ correction on the four corner cells, fitted
          by weighted maximum likelihood with an analytic gradient under L-BFGS-B.
        </p>

        <h2>2. Match weighting</h2>
        <p>
          Each match contributes <code>time_decay · importance</code>:
          exponential time decay with a half-life of 3,650 days (tuned via
          the WC 2022 sweep), multiplied by the eloratings.net K-factor
          (60 for WC finals down to 20 for friendlies).
        </p>

        <h2>3. Backtest gates</h2>
        <p>
          Calibration is judged by negative log-loss, Brier score, and Ranked
          Probability Score on a strict day-by-day hindcast. WC 2018 log-loss
          0.9585, WC 2022 log-loss 1.0379 — competitive with published
          bookmaker numbers.
        </p>

        <p className="text-xs text-muted-foreground">
          Full methodology lives at <code>docs/methodology.md</code> in the
          repository; this page sources it directly from there (Phase E adds
          the MDX sync step).
        </p>
      </section>
    </div>
  );
}
