"use client";

import { useMemo } from "react";
import { Group } from "@visx/group";
import { LinePath } from "@visx/shape";
import { scaleTime, scaleLinear } from "@visx/scale";
import { AxisBottom, AxisLeft } from "@visx/axis";
import { ParentSize } from "@visx/responsive";

const MARGIN = { top: 12, right: 16, bottom: 28, left: 44 };

type Point = { snapshot_date: string; rating: number };

type Props = {
  history: Point[];
  height?: number;
  color?: string;
};

/** Visx line chart of a team's Elo rating over time. */
export function EloLine({ history, height = 240, color = "#1f77b4" }: Props) {
  if (history.length === 0) return null;
  return (
    <div style={{ height }}>
      <ParentSize debounceTime={10}>
        {({ width }) => (
          <Inner width={width} height={height} history={history} color={color} />
        )}
      </ParentSize>
    </div>
  );
}

function Inner({
  width,
  height,
  history,
  color,
}: {
  width: number;
  height: number;
  history: Point[];
  color: string;
}) {
  const data = useMemo(
    () =>
      history.map((p) => ({
        date: new Date(p.snapshot_date + "T00:00:00Z").valueOf(),
        rating: p.rating,
      })),
    [history],
  );
  const innerW = Math.max(0, width - MARGIN.left - MARGIN.right);
  const innerH = Math.max(0, height - MARGIN.top - MARGIN.bottom);

  const xExtent = useMemo(
    () => [Math.min(...data.map((d) => d.date)), Math.max(...data.map((d) => d.date))],
    [data],
  );
  const yMin = Math.min(...data.map((d) => d.rating));
  const yMax = Math.max(...data.map((d) => d.rating));
  const yPad = Math.max(20, (yMax - yMin) * 0.1);
  const x = scaleTime({
    domain: [new Date(xExtent[0]), new Date(xExtent[1])],
    range: [0, innerW],
  });
  const y = scaleLinear({
    domain: [yMin - yPad, yMax + yPad],
    range: [innerH, 0],
    nice: true,
  });

  return (
    <svg width={width} height={height}>
      <Group left={MARGIN.left} top={MARGIN.top}>
        <AxisLeft
          scale={y}
          numTicks={4}
          stroke="currentColor"
          tickStroke="currentColor"
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
          numTicks={5}
          stroke="currentColor"
          tickStroke="currentColor"
          tickLabelProps={() => ({
            fontSize: 10,
            fill: "currentColor",
            textAnchor: "middle",
          })}
        />
        <LinePath
          data={data}
          x={(d) => x(d.date)}
          y={(d) => y(d.rating)}
          stroke={color}
          strokeWidth={1.5}
        />
        {data.map((d, i) => (
          <circle
            key={i}
            cx={x(d.date)}
            cy={y(d.rating)}
            r={2}
            fill={color}
            opacity={0.8}
          />
        ))}
      </Group>
    </svg>
  );
}
