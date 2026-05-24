"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

/**
 * In-page Match ID jumper for the 72 group-stage fixtures (0..71).
 * Pressing Enter or blurring with a valid id navigates without a full
 * page reload (the Server Component on the new route re-fetches).
 */
export function MatchIdInput({ current }: { current: number }) {
  const router = useRouter();
  const [value, setValue] = useState(String(current));

  function commit(raw: string) {
    const n = Number(raw);
    if (Number.isFinite(n) && n >= 0 && n <= 71 && n !== current) {
      router.push(`/match/${n}`);
    } else {
      setValue(String(current));
    }
  }

  return (
    <Label className="flex items-center gap-2 text-xs">
      <span className="text-muted-foreground">Match ID (0..71)</span>
      <Input
        type="number"
        min={0}
        max={71}
        step={1}
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onBlur={(e) => commit(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") commit((e.target as HTMLInputElement).value);
        }}
        className="w-20"
      />
    </Label>
  );
}
