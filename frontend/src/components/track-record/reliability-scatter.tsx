"use client";

import { Group } from "@visx/group";
import { LinePath } from "@visx/shape";
import { scaleLinear } from "@visx/scale";
import { AxisBottom, AxisLeft } from "@visx/axis";
import { ParentSize } from "@visx/responsive";

import { pct } from "@/lib/format";

const MARGIN = { top: 16, right: 16, bottom: 36, left: 48 };

export type ReliabilityBin = {
  outcome: "H" | "D" | "A";
  bin_low: number;
  bin_high: number;
  n: number;
  mean_predicted: number;
  realized_frequency: number;
};

const TRACE: Record<ReliabilityBin["outcome"], { color: string; label: string }> = {
  H: { color: "#1f77b4", label: "Home" },
  D: { color: "#7f7f7f", label: "Draw" },
  A: { color: "#d62728", label: "Away" },
};

export function ReliabilityScatter({
  bins,
  height = 420,
}: {
  bins: ReliabilityBin[];
  height?: number;
}) {
  return (
    <div style={{ height }}>
      <ParentSize debounceTime={10}>
        {({ width }) => <Inner width={width} height={height} bins={bins} />}
      </ParentSize>
    </div>
  );
}

function Inner({
  width,
  height,
  bins,
}: {
  width: number;
  height: number;
  bins: ReliabilityBin[];
}) {
  const innerW = Math.max(0, width - MARGIN.left - MARGIN.right);
  const innerH = Math.max(0, height - MARGIN.top - MARGIN.bottom);
  const x = scaleLinear({ domain: [0, 1], range: [0, innerW] });
  const y = scaleLinear({ domain: [0, 1], range: [innerH, 0] });

  const sorted = (out: ReliabilityBin["outcome"]) =>
    bins
      .filter((b) => b.outcome === out)
      .sort((a, b) => a.mean_predicted - b.mean_predicted);

  return (
    <svg width={width} height={height} role="img" aria-label="Reliability diagram">
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
          label="Realised frequency"
          labelOffset={36}
          labelProps={{
            fontSize: 11,
            fill: "currentColor",
            textAnchor: "middle",
          }}
        />
        <AxisBottom
          top={innerH}
          scale={x}
          numTicks={5}
          stroke="currentColor"
          tickStroke="currentColor"
          tickFormat={(v) => pct(Number(v), 0)}
          tickLabelProps={() => ({
            fontSize: 10,
            fill: "currentColor",
            textAnchor: "middle",
          })}
          label="Predicted probability"
          labelOffset={20}
          labelProps={{
            fontSize: 11,
            fill: "currentColor",
            textAnchor: "middle",
          }}
        />

        {/* y = x dashed reference line — perfect calibration. */}
        <line
          x1={x(0)}
          y1={y(0)}
          x2={x(1)}
          y2={y(1)}
          stroke="currentColor"
          strokeDasharray="4 4"
          strokeOpacity={0.4}
        />

        {(["H", "D", "A"] as const).map((out) => {
          const data = sorted(out);
          if (data.length === 0) return null;
          const { color, label } = TRACE[out];
          return (
            <g key={out}>
              <LinePath
                data={data}
                x={(d) => x(d.mean_predicted)}
                y={(d) => y(d.realized_frequency)}
                stroke={color}
                strokeWidth={1.5}
              />
              {data.map((d, i) => (
                <circle
                  key={i}
                  cx={x(d.mean_predicted)}
                  cy={y(d.realized_frequency)}
                  // Marker size scales with bin count so dense bins read louder.
                  r={Math.max(4, Math.min(10, Math.sqrt(d.n)))}
                  fill={color}
                  fillOpacity={0.7}
                >
                  <title>{`${label} ${pct(d.bin_low, 0)}-${pct(d.bin_high, 0)}: n=${d.n}, predicted ${pct(d.mean_predicted, 1)}, realised ${pct(d.realized_frequency, 1)}`}</title>
                </circle>
              ))}
              <text
                x={innerW - 4}
                y={16 * (["H", "D", "A"].indexOf(out) + 1)}
                fill={color}
                fontSize={11}
                textAnchor="end"
                fontWeight={600}
              >
                {label}
              </text>
            </g>
          );
        })}
      </Group>
    </svg>
  );
}
