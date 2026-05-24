"use client";

import { useMemo } from "react";
import { Group } from "@visx/group";
import { LinePath } from "@visx/shape";
import { scaleLinear } from "@visx/scale";
import { AxisBottom, AxisLeft } from "@visx/axis";
import { ParentSize } from "@visx/responsive";

import { pct } from "@/lib/format";
import type { LiveSnapshot } from "@/hooks/use-live-win-prob";

const MARGIN = { top: 16, right: 20, bottom: 32, left: 44 };
const COLORS = {
  home: "#1f77b4",
  draw: "#7f7f7f",
  away: "#d62728",
  goal: "#1f9d55",
  ft: "#888888",
};

/**
 * Stacked three-line live win-prob chart with vlines for every GOAL +
 * FT_WHISTLE event.
 */
export function LiveWinProbChart({
  events,
  homeTeam,
  awayTeam,
  height = 280,
}: {
  events: LiveSnapshot[];
  homeTeam: string;
  awayTeam: string;
  height?: number;
}) {
  if (events.length === 0) return null;
  return (
    <div style={{ height }}>
      <ParentSize debounceTime={10}>
        {({ width }) => (
          <Inner
            width={width}
            height={height}
            events={events}
            homeTeam={homeTeam}
            awayTeam={awayTeam}
          />
        )}
      </ParentSize>
    </div>
  );
}

function Inner({
  width,
  height,
  events,
  homeTeam,
  awayTeam,
}: {
  width: number;
  height: number;
  events: LiveSnapshot[];
  homeTeam: string;
  awayTeam: string;
}) {
  const innerW = Math.max(0, width - MARGIN.left - MARGIN.right);
  const innerH = Math.max(0, height - MARGIN.top - MARGIN.bottom);

  const xMax = useMemo(
    () => Math.max(95, ...events.map((e) => e.minute)),
    [events],
  );
  const x = scaleLinear({ domain: [0, xMax], range: [0, innerW] });
  const y = scaleLinear({ domain: [0, 1], range: [innerH, 0] });

  const goalEvents = events.filter((e) => e.last_event_type === "GOAL");
  const ftEvent = events.find((e) => e.last_event_type === "FT_WHISTLE");

  return (
    <svg width={width} height={height} role="img" aria-label="Live win probability">
      <Group left={MARGIN.left} top={MARGIN.top}>
        <AxisLeft
          scale={y}
          numTicks={5}
          stroke="currentColor"
          tickStroke="currentColor"
          tickFormat={(v) => pct(Number(v), 0)}
          tickLabelProps={() => ({
            fontSize: 10,
            fill: "currentColor",
            textAnchor: "end",
            dx: -4,
            dy: 3,
          })}
        />
        <AxisBottom
          top={innerH}
          scale={x}
          numTicks={Math.min(10, xMax)}
          stroke="currentColor"
          tickStroke="currentColor"
          label="minute"
          labelOffset={20}
          labelProps={{
            fontSize: 11,
            fill: "currentColor",
            textAnchor: "middle",
          }}
          tickLabelProps={() => ({
            fontSize: 10,
            fill: "currentColor",
            textAnchor: "middle",
          })}
        />

        {/* vlines for events */}
        {goalEvents.map((e, i) => (
          <line
            key={`goal-${i}`}
            x1={x(e.minute)}
            x2={x(e.minute)}
            y1={0}
            y2={innerH}
            stroke={COLORS.goal}
            strokeDasharray="4 3"
            strokeWidth={1}
          >
            <title>{`Goal at min ${e.minute}: ${e.home_score}-${e.away_score}`}</title>
          </line>
        ))}
        {ftEvent ? (
          <line
            x1={x(ftEvent.minute)}
            x2={x(ftEvent.minute)}
            y1={0}
            y2={innerH}
            stroke={COLORS.ft}
            strokeDasharray="2 3"
            strokeWidth={1}
          />
        ) : null}

        {/* Three lines */}
        {(
          [
            ["home_win", COLORS.home, homeTeam],
            ["draw", COLORS.draw, "Draw"],
            ["away_win", COLORS.away, awayTeam],
          ] as const
        ).map(([key, color, label]) => (
          <g key={key}>
            <LinePath
              data={events}
              x={(d) => x(d.minute)}
              y={(d) => y(d.win_prob[key])}
              stroke={color}
              strokeWidth={2}
            />
            <text
              x={innerW + 4}
              y={y(events[events.length - 1].win_prob[key])}
              fill={color}
              fontSize={10}
              dominantBaseline="central"
            >
              {label}
            </text>
          </g>
        ))}
      </Group>
    </svg>
  );
}
