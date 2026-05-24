import { apiGet, ApiError, ApiUnreachable } from "@/lib/api";
import { ApiUnreachableBanner } from "@/components/api-unreachable-banner";
import { MetricCard } from "@/components/metric-card";
import { JobTrigger } from "@/components/ops/job-trigger";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { HealthResponse } from "@/lib/types";

type SchedulerStatus = {
  jobs: {
    job_name: string;
    last_run_at: string | null;
    last_status: string | null;
    last_error_text: string | null;
  }[];
};

type AvailableJobs = { jobs: string[] };

export const metadata = { title: "Operator — WC 2026 Predictions" };

export default async function OperatorPage() {
  let health: HealthResponse | null = null;
  let status: SchedulerStatus | null = null;
  let avail: AvailableJobs | null = null;
  let unreachable = false;
  try {
    [health, status, avail] = await Promise.all([
      apiGet<HealthResponse>("/health", undefined, { noStore: true }),
      apiGet<SchedulerStatus>("/api/v1/_ops/scheduler-status", undefined, {
        noStore: true,
      }).catch(() => null),
      apiGet<AvailableJobs>("/api/v1/_ops/available-jobs", undefined, {
        revalidate: 600,
      }).catch(() => null),
    ]);
  } catch (err) {
    if (err instanceof ApiUnreachable) unreachable = true;
    else if (!(err instanceof ApiError)) throw err;
  }

  if (unreachable || !health) {
    return (
      <div className="space-y-6">
        <h1 className="ds-h1">Operator</h1>
        <ApiUnreachableBanner />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <header className="space-y-1">
        <h1 className="ds-h1">Operator</h1>
        <p className="text-xs text-muted-foreground">
          Operational health + manual scheduler-job triggers. Token validation
          happens server-side via the <code>WC2026_OPS_TOKEN</code> env var.
        </p>
      </header>

      <section className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <MetricCard
          label="Model fitted"
          value={health.model_fitted ? "yes" : "no"}
          help={health.model_version ?? "—"}
        />
        <MetricCard
          label="Model fit at"
          value={
            health.model_fit_at
              ? new Date(health.model_fit_at).toISOString().slice(0, 16) + "Z"
              : "—"
          }
        />
        <MetricCard
          label="Elo snapshot age"
          value={
            health.elo_snapshot_age_days != null
              ? `${health.elo_snapshot_age_days} days`
              : "—"
          }
          help={health.elo_snapshot_date ?? undefined}
        />
        <MetricCard
          label="Group letters"
          value={
            health.group_assignment_source.startsWith("official")
              ? "FIFA draw"
              : "derived"
          }
        />
      </section>

      <section className="space-y-2">
        <h2 className="text-lg font-semibold">Scheduler status</h2>
        {status ? (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Job</TableHead>
                <TableHead>Last run (UTC)</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Error</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {status.jobs.map((j) => (
                <TableRow key={j.job_name}>
                  <TableCell className="font-mono text-xs">{j.job_name}</TableCell>
                  <TableCell className="text-xs">{j.last_run_at ?? "—"}</TableCell>
                  <TableCell className="text-xs">
                    {j.last_status === "ok" ? (
                      <span className="text-emerald-500">ok</span>
                    ) : j.last_status ? (
                      <span className="text-destructive">{j.last_status}</span>
                    ) : (
                      "—"
                    )}
                  </TableCell>
                  <TableCell className="text-xs text-destructive">
                    {j.last_error_text ?? ""}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        ) : (
          <p className="text-xs text-muted-foreground">
            Scheduler status unavailable (Postgres may be down or the API
            doesn&apos;t expose <code>/_ops/scheduler-status</code>).
          </p>
        )}
      </section>

      <section className="space-y-2">
        <h2 className="text-lg font-semibold">Manual triggers</h2>
        {avail && avail.jobs.length > 0 ? (
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
            {avail.jobs.map((name) => (
              <JobTrigger key={name} name={name} />
            ))}
          </div>
        ) : (
          <p className="text-xs text-muted-foreground">
            No manually-triggerable jobs registered.
          </p>
        )}
      </section>
    </div>
  );
}
