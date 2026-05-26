"use client";

import Image from "next/image";
import Link from "next/link";
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
  /**
   * Render the chip as a `<Link>` to `/team/[name]`. Only opt in when the
   * chip is NOT already nested inside another `<a>` / `<Link>` — nested
   * anchors are invalid HTML and Next will warn at runtime.
   */
  asLink?: boolean;
};

/**
 * Inline crest + team name. Falls back gracefully when no asset row exists
 * (the hook returns an all-null payload). Used everywhere a team appears in
 * the dashboard (Today cards, Match Detail header, Groups table, etc.).
 */
export function TeamChip({
  team,
  size = "md",
  bold,
  className,
  asLink = false,
}: TeamChipProps) {
  const { data } = useTeamAssets(team);
  const { container, img, text } = SIZE_CLASSES[size];
  const crest = data?.crest_url;
  const inner = (
    <>
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
    </>
  );
  if (asLink) {
    return (
      <Link
        href={`/team/${encodeURIComponent(team)}`}
        className={cn(
          "inline-flex items-center rounded-sm transition-colors hover:text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
          container,
          className,
        )}
        aria-label={`Open ${team} profile`}
      >
        {inner}
      </Link>
    );
  }
  return (
    <span className={cn("inline-flex items-center", container, className)}>
      {inner}
    </span>
  );
}

/**
 * "Home vs Away" header with both crests, used on the Match Detail page.
 * Each chip links to the team profile.
 */
export function VersusHeader({ home, away }: { home: string; away: string }) {
  return (
    <div className="flex items-center gap-3 text-xl">
      <TeamChip team={home} size="lg" bold asLink />
      <span className="text-muted-foreground text-base">vs</span>
      <TeamChip team={away} size="lg" bold asLink />
    </div>
  );
}
