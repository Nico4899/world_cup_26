"use client";

import { useRouter } from "next/navigation";

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

export function TeamPicker({
  current,
  teams,
}: {
  current: string;
  teams: string[];
}) {
  const router = useRouter();
  return (
    <Select
      value={current}
      onValueChange={(v) => {
        if (v) router.push(`/team/${encodeURIComponent(v)}`);
      }}
    >
      <SelectTrigger className="w-72">
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {teams.map((t) => (
          <SelectItem key={t} value={t}>
            {t}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}
