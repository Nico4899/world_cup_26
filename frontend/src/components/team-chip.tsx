"use client";

import Image from "next/image";
import { cn } from "@/lib/utils";
import { useTeamAssets } from "@/hooks/use-team-assets";

type Size = "sm" | "md" | "lg";

const SIZE_CLASSES: Record<Size, { container: string; img: number; text: string }> = {
  sm: { container: "gap-1.5 text-sm", img: 18, text: "" },
  md: { container: "gap-2 text-base", img: 24, text: "" },
  lg: { container: "gap-3 text-xl", img: 40, text: "font-semibold" },
};

type TeamChipProps = {
  team: string;
  size?: Size;
  bold?: boolean;
  className?: string;
};

/**
 * Inline crest + team name. Falls back gracefully when no asset row exists
 * (the hook returns an all-null payload). Used everywhere a team appears in
 * the dashboard (Today cards, Match Detail header, Groups table, etc.).
 */
export function TeamChip({ team, size = "md", bold, className }: TeamChipProps) {
  const { data } = useTeamAssets(team);
  const { container, img, text } = SIZE_CLASSES[size];
  const crest = data?.crest_url;
  return (
    <span className={cn("inline-flex items-center", container, className)}>
      {crest ? (
        <Image
          src={crest}
          alt=""
          width={img}
          height={img}
          unoptimized
          className="rounded-sm"
        />
      ) : null}
      <span className={cn(text, bold && "font-semibold")}>{team}</span>
    </span>
  );
}

/**
 * "Home vs Away" header with both crests, used on the Match Detail page.
 */
export function VersusHeader({ home, away }: { home: string; away: string }) {
  return (
    <div className="flex items-center gap-3 text-xl">
      <TeamChip team={home} size="lg" bold />
      <span className="text-muted-foreground text-base">vs</span>
      <TeamChip team={away} size="lg" bold />
    </div>
  );
}
