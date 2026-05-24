import { apiGet, ApiError, ApiUnreachable } from "@/lib/api";
import { Card, CardContent } from "@/components/ui/card";
import { MethodologyToC } from "@/components/methodology-toc";
import Methodology from "@/content/methodology.mdx";
import type { HealthResponse } from "@/lib/types";

export const metadata = {
  title: "About / Methodology — WC 2026 Predictions",
};

/**
 * Server-rendered About page.
 *
 * The methodology content is imported as MDX (compiled by @next/mdx at build
 * time) from `src/content/methodology.mdx`, which is itself synced from
 * `docs/methodology.md` by `scripts/sync-methodology.mjs` (wired into
 * `pnpm predev` + `pnpm prebuild`). The page also surfaces the
 * group-letter assignment source from `/health`.
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
    <div className="space-y-6">
      <header>
        <h1 className="ds-h1">About / Methodology</h1>
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

      <div className="grid lg:grid-cols-[1fr_220px] gap-8">
        <article className="prose prose-sm dark:prose-invert max-w-3xl">
          <Methodology />
        </article>
        <aside>
          <MethodologyToC />
        </aside>
      </div>
    </div>
  );
}
