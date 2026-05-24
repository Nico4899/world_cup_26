"use client";

import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";

import { ApiUnreachable, apiGet } from "@/lib/api";
import { ApiUnreachableBanner } from "@/components/api-unreachable-banner";
import { HOST_CITIES } from "@/lib/host-cities";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { FixtureSummary } from "@/lib/types";

const ALL = "(all)";

export function FixtureList() {
  const [choice, setChoice] = useState<string>(ALL);
  const { data, error } = useQuery({
    queryKey: ["matches-list"],
    queryFn: () => apiGet<FixtureSummary[]>("/api/v1/matches"),
    staleTime: 5 * 60 * 1000,
  });

  const cities = useMemo(() => {
    const set = new Set<string>(HOST_CITIES.map((c) => c.city));
    (data ?? []).forEach((m) => set.add(m.city));
    return [ALL, ...Array.from(set).sort()];
  }, [data]);

  if (error instanceof ApiUnreachable) return <ApiUnreachableBanner />;
  if (!data) {
    return <p className="text-sm text-muted-foreground">Loading fixtures…</p>;
  }
  const filtered = choice === ALL ? data : data.filter((m) => m.city === choice);

  return (
    <div className="space-y-3">
      <label className="flex flex-col gap-1 text-sm max-w-md">
        <span className="text-xs uppercase tracking-wide text-muted-foreground">
          Filter fixtures by host city
        </span>
        <Select value={choice} onValueChange={(v) => setChoice(v ?? ALL)}>
          <SelectTrigger>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {cities.map((c) => (
              <SelectItem key={c} value={c}>
                {c}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </label>
      <h2 className="text-lg font-semibold">
        {choice === ALL
          ? "All 72 fixtures"
          : `${filtered.length} fixture${filtered.length === 1 ? "" : "s"} at ${choice}`}
      </h2>
      {filtered.length === 0 ? (
        <p className="text-sm text-muted-foreground">
          No fixtures recorded at {choice}. The dataset&apos;s city label may
          differ — try the dropdown alternatives.
        </p>
      ) : (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Match #</TableHead>
              <TableHead>Date</TableHead>
              <TableHead>Group</TableHead>
              <TableHead>Home</TableHead>
              <TableHead>Away</TableHead>
              <TableHead>City</TableHead>
              <TableHead>Country</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {filtered.map((m) => (
              <TableRow key={m.match_id}>
                <TableCell className="text-xs tabular-nums">{m.match_id}</TableCell>
                <TableCell className="text-xs">{m.date}</TableCell>
                <TableCell className="text-xs">{m.group}</TableCell>
                <TableCell className="text-xs">{m.home_team}</TableCell>
                <TableCell className="text-xs">{m.away_team}</TableCell>
                <TableCell className="text-xs">{m.city}</TableCell>
                <TableCell className="text-xs">{m.country}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      )}
    </div>
  );
}
