"use client";

import { useMemo } from "react";
import { ParentSize } from "@visx/responsive";
import { Group } from "@visx/group";
import { scaleLinear } from "@visx/scale";

import { cn } from "@/lib/utils";

export type ProbabilitySegment = {
  /** Display label (rendered in legend; not inside the bar). */
  label: string;
  /** Probability in [0, 1]. Segments may not sum to exactly 1.0; the bar
   *  renders them faithfully without normalising. */
  value: number;
  /** Solid fill colour (hex, hsl, oklch, etc.). */
  color: string;
};

type Props = {
  segments: ProbabilitySegment[];
  /** Fixed pixel height; width is responsive. */
  height?: number;
  /** Show inline percentages on segments wider than 5% of the bar. */
  showLabels?: boolean;
  className?: string;
};

/**
 * Horizontal stacked probability bar built directly from `<rect>` so the
 * dashboard can use the same primitive for the 1X2 outcome bar (3 segments),
 * the group-stage stack (5 segments: 1st / 2nd / 3rd-adv / 3rd-out / 4th),
 * and the Today summary strip. Replaces the Streamlit Plotly Bar+stack.
 */
export function ProbabilityBar({
  segments,
  height = 36,
  showLabels = true,
  className,
}: Props) {
  return (
    <div className={cn("w-full", className)} style={{ height }}>
      <ParentSize debounceTime={10}>
        {({ width }) => (
          <ProbabilityBarInner
            width={width}
            height={height}
            segments={segments}
            showLabels={showLabels}
          />
        )}
      </ParentSize>
    </div>
  );
}

function ProbabilityBarInner({
  width,
  height,
  segments,
  showLabels,
}: {
  width: number;
  height: number;
  segments: ProbabilitySegment[];
  showLabels: boolean;
}) {
  // Total may not be exactly 1 (e.g. when a 3rd-place team appears in both
  // advance and elimination tallies); normalise to keep the bar full-width.
  const total = useMemo(
    () => segments.reduce((s, x) => s + Math.max(x.value, 0), 0) || 1,
    [segments],
  );
  const scale = scaleLinear({ domain: [0, total], range: [0, width] });

  let acc = 0;
  return (
    <svg width={width} height={height} role="img" aria-label="Probability bar">
      <Group>
        {segments.map((seg, i) => {
          const v = Math.max(seg.value, 0);
          const x = scale(acc);
          const w = scale(acc + v) - x;
          acc += v;
          const pct = v / total;
          const visible = showLabels && pct >= 0.05;
          return (
            <g key={`${seg.label}-${i}`}>
              <rect
                x={x}
                y={0}
                width={Math.max(w, 0)}
                height={height}
                fill={seg.color}
              >
                <title>{`${seg.label}: ${(pct * 100).toFixed(1)}%`}</title>
              </rect>
              {visible ? (
                <text
                  x={x + w / 2}
                  y={height / 2}
                  textAnchor="middle"
                  dominantBaseline="central"
                  className="fill-white text-[10px] font-semibold pointer-events-none"
                >
                  {`${Math.round(pct * 100)}%`}
                </text>
              ) : null}
            </g>
          );
        })}
      </Group>
    </svg>
  );
}

export function ProbabilityLegend({ segments }: { segments: ProbabilitySegment[] }) {
  return (
    <div className="flex flex-wrap gap-x-3 gap-y-1 text-xs text-muted-foreground">
      {segments.map((seg) => (
        <span key={seg.label} className="inline-flex items-center gap-1.5">
          <span
            className="inline-block h-2.5 w-2.5 rounded-sm"
            style={{ background: seg.color }}
            aria-hidden
          />
          {seg.label}
        </span>
      ))}
    </div>
  );
}
