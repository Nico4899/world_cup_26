"use client";

import { ChevronRight } from "lucide-react";
import type { ReactNode } from "react";

import { cn } from "@/lib/utils";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";

type MetricCardProps = {
  /** Top-line label (e.g. "Argentina", "Champion"). */
  label: string;
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
  const body = (
    <div
      className={cn(
        "rounded-lg border bg-card text-card-foreground p-4 flex flex-col gap-1",
        popover &&
          "cursor-pointer hover:bg-accent/40 transition-colors text-left w-full",
        className,
      )}
    >
      <div className="flex items-start justify-between gap-2">
        <p className="text-xs uppercase tracking-wide text-muted-foreground">
          {label}
        </p>
        {popover ? (
          <ChevronRight className="h-3.5 w-3.5 text-muted-foreground" aria-hidden />
        ) : null}
      </div>
      <p className="text-2xl font-semibold leading-tight">{value}</p>
      {delta ? (
        <p className="text-xs text-muted-foreground">{delta}</p>
      ) : null}
      {help ? <p className="text-xs text-muted-foreground">{help}</p> : null}
    </div>
  );

  if (!popover) return body;

  return (
    <Popover>
      <PopoverTrigger className="text-left rounded-lg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring">
        {body}
      </PopoverTrigger>
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
