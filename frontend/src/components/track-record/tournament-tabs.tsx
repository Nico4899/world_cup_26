"use client";

import { useRouter, useSearchParams, usePathname } from "next/navigation";

import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";

const TOURNAMENTS = ["WC2022", "WC2018"] as const;

/**
 * URL-bound tournament toggle for the Track Record historical section.
 */
export function TournamentTabs({ current }: { current: (typeof TOURNAMENTS)[number] }) {
  const router = useRouter();
  const pathname = usePathname();
  const params = useSearchParams();
  return (
    <Tabs
      value={current}
      onValueChange={(v) => {
        const next = new URLSearchParams(params.toString());
        next.set("tournament", v);
        router.replace(`${pathname}?${next.toString()}`);
      }}
    >
      <TabsList>
        {TOURNAMENTS.map((t) => (
          <TabsTrigger key={t} value={t}>
            {t === "WC2022" ? "WC 2022" : "WC 2018"}
          </TabsTrigger>
        ))}
      </TabsList>
    </Tabs>
  );
}
