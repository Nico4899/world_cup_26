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

function blueOf(t: number): string {
  // 0..1 → light → deep blue; matches Streamlit's "Blues" cmap roughly.
  const lerp = (a: number, b: number) => Math.round(a + (b - a) * t);
  return `rgb(${lerp(247, 8)}, ${lerp(251, 81)}, ${lerp(255, 156)})`;
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
              const fill = blueOf(intensity(v));
              const labelOk = v >= 0.02;
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
                      className="text-[9px] fill-current pointer-events-none tabular-nums"
                      style={{ color: intensity(v) > 0.55 ? "white" : "#1a365d" }}
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
