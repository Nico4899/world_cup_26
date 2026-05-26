"use client";

import { ChevronRight } from "lucide-react";
import type { ReactNode } from "react";

import { cn } from "@/lib/utils";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";

type MetricCardProps = {
  /** Top-line label. Accepts ReactNode so callers can append a
   *  `<HelpDot/>` for inline glossary lookup. */
  label: ReactNode;
  /** Big value (e.g. "48.2%", "192", "+0.74"). */
  value: string;
  /** Optional small caption below the value. */
  help?: string;
  /** Optional change indicator displayed under the value. */
  delta?: string;
  className?: string;
  /** When provided, the whole tile becomes a popover trigger; clicking
   *  opens the popover with the supplied node ("what drives this" panel). */
  popover?: ReactNode;
  /** Popover title shown above the body. */
  popoverTitle?: string;
};

/**
 * Labelled big-number tile. The popover variant satisfies the spec's
 * "every probability shown must be clickable to reveal a 'what drives this'
 * panel" principle — clicking the tile opens a Radix popover.
 */
export function MetricCard({
  label,
  value,
  help,
  delta,
  className,
  popover,
  popoverTitle,
}: MetricCardProps) {
  if (!popover) {
    return (
      <div
        className={cn(
          "rounded-lg border bg-card text-card-foreground p-4 flex flex-col gap-1",
          className,
        )}
      >
        <div className="flex items-start justify-between gap-2">
          <div className="text-xs uppercase tracking-wide text-muted-foreground inline-flex items-center gap-1">
            {label}
          </div>
        </div>
        <p className="ds-metric">{value}</p>
        {delta ? (
          <p className="text-xs text-muted-foreground">{delta}</p>
        ) : null}
        {help ? <p className="text-xs text-muted-foreground">{help}</p> : null}
      </div>
    );
  }

  return (
    <Popover>
      <div
        className={cn(
          "relative rounded-lg border bg-card text-card-foreground p-4 flex flex-col gap-1 transition-colors hover:bg-accent/40 has-[button:focus-visible]:ring-2 has-[button:focus-visible]:ring-ring",
          className,
        )}
      >
        <PopoverTrigger
          type="button"
          aria-label={
            popoverTitle ? `Open ${popoverTitle} details` : "Open details"
          }
          className="absolute inset-0 z-0 rounded-lg cursor-pointer focus-visible:outline-none"
        />
        <div className="relative z-10 pointer-events-none flex items-start justify-between gap-2">
          <div className="text-xs uppercase tracking-wide text-muted-foreground inline-flex items-center gap-1 [&_button]:pointer-events-auto">
            {label}
          </div>
          <ChevronRight
            className="h-3.5 w-3.5 text-muted-foreground"
            aria-hidden
          />
        </div>
        <p className="relative z-10 pointer-events-none ds-metric">{value}</p>
        {delta ? (
          <p className="relative z-10 pointer-events-none text-xs text-muted-foreground">
            {delta}
          </p>
        ) : null}
        {help ? (
          <p className="relative z-10 pointer-events-none text-xs text-muted-foreground">
            {help}
          </p>
        ) : null}
      </div>
      <PopoverContent className="w-80 space-y-2">
        {popoverTitle ? (
          <p className="text-sm font-medium">{popoverTitle}</p>
        ) : null}
        <div className="text-sm text-muted-foreground space-y-1.5">
          {popover}
        </div>
      </PopoverContent>
    </Popover>
  );
}
