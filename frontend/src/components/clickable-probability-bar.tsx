"use client";

import {
  ProbabilityBar,
  type ProbabilitySegment,
} from "@/components/probability-bar";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { pct } from "@/lib/format";

type Props = {
  /** Heading rendered at the top of the popover (e.g. team name). */
  title: string;
  segments: ProbabilitySegment[];
  /** Optional caption rendered under the segment list inside the popover. */
  caption?: string;
  height?: number;
};

/**
 * Clickable wrapper around `<ProbabilityBar>`. Each segment's exact
 * percentage shows in the popover body, plus an optional caption (used
 * by the Bracket page to surface the MC provenance / n_sims).
 *
 * The bar itself is still hoverable for the per-segment SVG `<title>`
 * tooltips; the popover adds the spec-mandated "every probability shown
 * must be clickable to reveal a 'what drives this' panel" affordance.
 */
export function ClickableProbabilityBar({
  title,
  segments,
  caption,
  height = 20,
}: Props) {
  const total = segments.reduce((s, x) => s + Math.max(x.value, 0), 0) || 1;
  return (
    <Popover>
      <PopoverTrigger className="w-full text-left rounded focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring">
        <ProbabilityBar segments={segments} height={height} />
      </PopoverTrigger>
      <PopoverContent className="w-72 space-y-2">
        <p className="text-sm font-medium">{title}</p>
        <ul className="space-y-1 text-xs">
          {segments.map((seg) => (
            <li key={seg.label} className="flex items-center gap-2">
              <span
                aria-hidden
                className="inline-block h-2.5 w-2.5 rounded-sm shrink-0"
                style={{ background: seg.color }}
              />
              <span className="flex-1">{seg.label}</span>
              <span className="tabular-nums font-medium">
                {pct(seg.value / total)}
              </span>
            </li>
          ))}
        </ul>
        {caption ? (
          <p className="text-[11px] text-muted-foreground border-t pt-2">
            {caption}
          </p>
        ) : null}
      </PopoverContent>
    </Popover>
  );
}
