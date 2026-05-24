"use client";

import { useMemo } from "react";
import { Group } from "@visx/group";
import { scaleLinear } from "@visx/scale";

const DISPLAY_GOALS = 5; // Crop the 11x11 matrix to 0..5 per the spec.
const MARGIN = { top: 28, right: 12, bottom: 36, left: 36 };

type Props = {
  matrix: number[][]; // [home][away] joint probability
  homeTeam: string;
  awayTeam: string;
  width?: number;
};

// Viridis ramp (poster-palette v2 default). Hard-coded — mirrors the
// --heat-stop-N CSS variables in globals.css; keep the two in sync.
const VIRIDIS_STOPS: ReadonlyArray<readonly [number, number, number]> = [
  [68, 1, 84], // #440154
  [59, 82, 139], // #3b528b
  [33, 144, 141], // #21908d
  [93, 201, 99], // #5dc963
  [253, 231, 37], // #fde725
];

function viridisOf(t: number): string {
  // 0..1 → purple → teal → yellow, piecewise-linear across 5 stops.
  const clamped = Math.max(0, Math.min(1, t));
  const span = clamped * (VIRIDIS_STOPS.length - 1);
  const i = Math.min(VIRIDIS_STOPS.length - 2, Math.floor(span));
  const f = span - i;
  const a = VIRIDIS_STOPS[i];
  const b = VIRIDIS_STOPS[i + 1];
  const lerp = (x: number, y: number) => Math.round(x + (y - x) * f);
  return `rgb(${lerp(a[0], b[0])}, ${lerp(a[1], b[1])}, ${lerp(a[2], b[2])})`;
}

export function ProbHeatmap({ matrix, homeTeam, awayTeam, width = 380 }: Props) {
  const crop = useMemo(
    () =>
      matrix
        .slice(0, DISPLAY_GOALS + 1)
        .map((row) => row.slice(0, DISPLAY_GOALS + 1)),
    [matrix],
  );
  const cropSum = useMemo(
    () => crop.reduce((a, row) => a + row.reduce((b, v) => b + v, 0), 0),
    [crop],
  );
  const totalSum = useMemo(
    () => matrix.reduce((a, row) => a + row.reduce((b, v) => b + v, 0), 0),
    [matrix],
  );
  const truncatedMass = Math.max(0, totalSum - cropSum);

  const max = useMemo(
    () => crop.reduce((a, row) => Math.max(a, ...row), 0) || 1,
    [crop],
  );

  const innerW = width - MARGIN.left - MARGIN.right;
  const cell = innerW / (DISPLAY_GOALS + 1);
  const innerH = cell * (DISPLAY_GOALS + 1);
  const height = innerH + MARGIN.top + MARGIN.bottom;
  const intensity = scaleLinear({ domain: [0, max], range: [0, 1] });

  return (
    <div className="space-y-2">
      <p className="text-xs text-muted-foreground">
        Joint score probability (0-{DISPLAY_GOALS} shown
        {truncatedMass > 0.005
          ? `; ${(truncatedMass * 100).toFixed(1)}% mass beyond`
          : ""}
        )
      </p>
      <svg width={width} height={height} role="img" aria-label="Score heatmap">
        <text
          x={width / 2}
          y={MARGIN.top - 12}
          textAnchor="middle"
          className="text-[11px] fill-current text-muted-foreground"
        >
          {awayTeam} goals →
        </text>
        <Group
          left={MARGIN.left}
          top={MARGIN.top}
          transform={`translate(${MARGIN.left}, ${MARGIN.top}) rotate(-90)`}
        >
          <text
            x={-innerH / 2}
            y={-22}
            textAnchor="middle"
            className="text-[11px] fill-current text-muted-foreground"
          >
            {homeTeam} goals
          </text>
        </Group>
        <Group left={MARGIN.left} top={MARGIN.top}>
          {/* Column headers */}
          {Array.from({ length: DISPLAY_GOALS + 1 }, (_, j) => (
            <text
              key={`col-${j}`}
              x={j * cell + cell / 2}
              y={-4}
              textAnchor="middle"
              className="text-[10px] fill-current text-muted-foreground tabular-nums"
            >
              {j}
            </text>
          ))}
          {/* Row headers */}
          {Array.from({ length: DISPLAY_GOALS + 1 }, (_, i) => (
            <text
              key={`row-${i}`}
              x={-6}
              y={i * cell + cell / 2}
              dominantBaseline="central"
              textAnchor="end"
              className="text-[10px] fill-current text-muted-foreground tabular-nums"
            >
              {i}
            </text>
          ))}
          {/* Cells */}
          {crop.map((row, i) =>
            row.map((v, j) => {
              const t = intensity(v);
              const fill = viridisOf(t);
              const labelOk = v >= 0.02;
              // Viridis is darkest at the ends — use dark label only on the
              // bright teal/green/yellow band, else white.
              const labelColor = t > 0.45 && t < 0.95 ? "#0a0a0a" : "#fafafa";
              return (
                <g key={`${i}-${j}`}>
                  <rect
                    x={j * cell}
                    y={i * cell}
                    width={cell}
                    height={cell}
                    fill={fill}
                    stroke="white"
                    strokeWidth={0.5}
                  >
                    <title>{`${homeTeam} ${i} - ${j} ${awayTeam}: ${(v * 100).toFixed(2)}%`}</title>
                  </rect>
                  {labelOk ? (
                    <text
                      x={j * cell + cell / 2}
                      y={i * cell + cell / 2}
                      textAnchor="middle"
                      dominantBaseline="central"
                      className="text-[9px] pointer-events-none tabular-nums"
                      fill={labelColor}
                    >
                      {`${(v * 100).toFixed(1)}%`}
                    </text>
                  ) : null}
                </g>
              );
            }),
          )}
        </Group>
      </svg>
    </div>
  );
}
