"use server";

import { apiPost, API_URL } from "@/lib/api";

type RunJobResult = {
  ok: boolean;
  status?: string;
  enqueuedAt?: string;
  error?: string;
};

/**
 * Server Action that triggers a manual scheduler job.
 *
 * The `WC2026_OPS_TOKEN` is read from server-only env (never reaches the
 * browser) and attached as `X-Ops-Token`. The FastAPI op route checks it
 * against its own env-loaded token; the operator just clicks the button.
 */
export async function runJob(jobName: string): Promise<RunJobResult> {
  const token = process.env.WC2026_OPS_TOKEN;
  try {
    const data = await apiPost<{ job_name: string; enqueued_at: string; status: string }>(
      `/api/v1/_ops/run-job/${encodeURIComponent(jobName)}`,
      {},
      { headers: token ? { "X-Ops-Token": token } : undefined, noStore: true },
    );
    return { ok: true, status: data.status, enqueuedAt: data.enqueued_at };
  } catch (err) {
    const msg = err instanceof Error ? err.message : "unknown error";
    return { ok: false, error: `${API_URL}: ${msg}` };
  }
}
