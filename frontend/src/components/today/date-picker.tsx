"use client";

import { useRouter, useSearchParams, usePathname } from "next/navigation";
import { useCallback } from "react";

import { Input } from "@/components/ui/input";

const WC_START = "2026-06-11";
const WC_END = "2026-07-19";

/**
 * URL-bound matchday picker. Selecting a new date replaces ?date=... in the
 * URL, triggering a Server Component re-fetch of /api/v1/matches.
 */
export function DatePicker({ initial }: { initial: string }) {
  const router = useRouter();
  const pathname = usePathname();
  const params = useSearchParams();
  const onChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const next = new URLSearchParams(params.toString());
      next.set("date", e.target.value);
      router.replace(`${pathname}?${next.toString()}`);
    },
    [router, pathname, params],
  );
  return (
    <label className="flex items-center gap-2 text-sm">
      <span className="text-muted-foreground">Matchday</span>
      <Input
        type="date"
        defaultValue={initial}
        min={WC_START}
        max={WC_END}
        onChange={onChange}
        className="w-44"
      />
    </label>
  );
}
