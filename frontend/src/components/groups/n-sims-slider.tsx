"use client";

import { useRouter, useSearchParams, usePathname } from "next/navigation";
import { useState } from "react";

import { Slider } from "@/components/ui/slider";
import { HelpDot } from "@/components/help-dot";

type Props = { initial: number };

/**
 * URL-bound Monte Carlo n_sims slider for the Groups page.
 *
 * Slider drag is local state; release commits to ?n_sims= via
 * `router.replace` so the Server Component re-fetches /standings.
 */
export function NSimsSlider({ initial }: Props) {
  const router = useRouter();
  const pathname = usePathname();
  const params = useSearchParams();
  const [value, setValue] = useState(initial);

  function commit(next: number) {
    const url = new URLSearchParams(params.toString());
    url.set("n_sims", String(next));
    router.replace(`${pathname}?${url.toString()}`);
  }

  return (
    <div className="space-y-2">
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-xs uppercase tracking-wide text-muted-foreground inline-flex items-center">
          Monte Carlo simulations
          <HelpDot term="Monte Carlo" />
        </span>
        <span className="text-sm font-medium tabular-nums">
          {value.toLocaleString()}
        </span>
      </div>
      <Slider
        min={200}
        max={10_000}
        step={200}
        value={[value]}
        onValueChange={(v) => {
          const next = Array.isArray(v) ? v[0] : v;
          if (typeof next === "number") setValue(next);
        }}
        onValueCommitted={(v) => {
          const next = Array.isArray(v) ? v[0] : v;
          if (typeof next === "number") commit(next);
        }}
      />
    </div>
  );
}
