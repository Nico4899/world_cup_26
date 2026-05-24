"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { useTheme } from "next-themes";
import {
  CalendarClock,
  Download,
  GitBranch,
  Info,
  LayoutGrid,
  LineChart,
  MapPinned,
  Moon,
  RefreshCw,
  Sparkles,
  Sun,
  Terminal,
  Trophy,
  Users,
  type LucideIcon,
} from "lucide-react";

import { apiGet } from "@/lib/api";
import type { FixtureSummary } from "@/lib/types";
import { cn } from "@/lib/utils";

type Item = {
  id: string;
  label: string;
  meta: string;
  Icon: LucideIcon;
  href?: string;
  onSelect?: () => void;
};

type Props = {
  open: boolean;
  onClose: () => void;
};

const ROUTE_ITEMS: Item[] = [
  { id: "today", label: "Today", meta: "matchday cards", Icon: CalendarClock, href: "/" },
  { id: "match", label: "Match Detail", meta: "match #0 by default", Icon: Trophy, href: "/match/0" },
  { id: "groups", label: "Groups", meta: "12 groups", Icon: LayoutGrid, href: "/groups" },
  { id: "bracket", label: "Bracket", meta: "32-team knockout", Icon: GitBranch, href: "/bracket" },
  { id: "track", label: "Track Record", meta: "calibration receipts", Icon: LineChart, href: "/track-record" },
  { id: "team", label: "Team Profile", meta: "Argentina by default", Icon: Users, href: "/team/Argentina" },
  { id: "map", label: "Map", meta: "16 host venues", Icon: MapPinned, href: "/map" },
  { id: "about", label: "About", meta: "methodology + lineage", Icon: Info, href: "/about" },
  { id: "ops", label: "Operator", meta: "scheduler / health", Icon: Terminal, href: "/ops" },
];

/**
 * Global ⌘K palette. Three groups: Routes (all 9), Teams (sourced live
 * from /matches), Actions (theme toggle + a few placeholder no-ops). Open
 * with ⌘K / Ctrl+K, Esc closes, ↑/↓ moves selection, ↵ activates.
 */
export function CommandPalette({ open, onClose }: Props) {
  const router = useRouter();
  const { resolvedTheme, setTheme } = useTheme();
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const { data: matches } = useQuery({
    queryKey: ["matches-list"],
    queryFn: () => apiGet<FixtureSummary[]>("/api/v1/matches"),
    staleTime: 5 * 60 * 1000,
    retry: false,
    enabled: open,
  });

  const teams = useMemo(() => {
    if (!matches) return [] as string[];
    const set = new Set<string>();
    for (const m of matches) {
      set.add(m.home_team);
      set.add(m.away_team);
    }
    return Array.from(set).sort();
  }, [matches]);

  const groups = useMemo(() => {
    const teamItems: Item[] = teams.map((t) => ({
      id: `team-${t}`,
      label: t,
      meta: "open team profile",
      Icon: Users,
      href: `/team/${encodeURIComponent(t)}`,
    }));
    const actionItems: Item[] = [
      {
        id: "theme-toggle",
        label: "Toggle theme",
        meta: resolvedTheme === "dark" ? "to light" : "to dark",
        Icon: resolvedTheme === "dark" ? Sun : Moon,
        onSelect: () => setTheme(resolvedTheme === "dark" ? "light" : "dark"),
      },
      { id: "act-shap", label: "Search SHAP top features", meta: "explain endpoint", Icon: Sparkles },
      { id: "act-dl", label: "Download bracket as PNG", meta: "html-to-image", Icon: Download },
      { id: "act-refit", label: "Force model refit", meta: "ops · destructive", Icon: RefreshCw },
    ];
    const q = query.trim().toLowerCase();
    const filt = (items: Item[]) =>
      !q
        ? items
        : items.filter(
            (it) =>
              it.label.toLowerCase().includes(q) || it.meta.toLowerCase().includes(q),
          );
    return [
      { title: "Routes", items: filt(ROUTE_ITEMS) },
      { title: "Teams", items: filt(teamItems) },
      { title: "Actions", items: filt(actionItems) },
    ].filter((g) => g.items.length > 0);
  }, [query, teams, resolvedTheme, setTheme]);

  const flat = useMemo(() => groups.flatMap((g) => g.items), [groups]);
  const safeActive = Math.min(active, Math.max(0, flat.length - 1));

  // Reset input + focus when the palette opens. State resets here are the
  // canonical pattern (cf. theme-toggle.tsx) — the alternative `key` reset
  // would unmount the dialog mid-keystroke.
  useEffect(() => {
    if (!open) return;
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setQuery("");
    setActive(0);
    const t = setTimeout(() => inputRef.current?.focus(), 10);
    return () => clearTimeout(t);
  }, [open]);

  // Keyboard handlers while open.
  useEffect(() => {
    if (!open) return;
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
        return;
      }
      if (e.key === "ArrowDown") {
        e.preventDefault();
        setActive((a) => Math.min(flat.length - 1, a + 1));
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        setActive((a) => Math.max(0, a - 1));
      } else if (e.key === "Enter") {
        e.preventDefault();
        const sel = flat[safeActive];
        if (sel) {
          if (sel.href) router.push(sel.href);
          else sel.onSelect?.();
          onClose();
        }
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, flat, safeActive, router, onClose]);

  if (!open) return null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="Command palette"
      className="fixed inset-0 z-50 flex justify-center pt-30 bg-primary/20 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="flex w-[min(620px,calc(100%-2rem))] max-h-[calc(100vh-200px)] flex-col overflow-hidden rounded-xl border bg-card shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center gap-3 border-b px-5 py-4">
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setActive(0);
            }}
            placeholder="Jump to anything — teams, matches, routes, actions…"
            className="flex-1 bg-transparent text-sm outline-none placeholder:text-muted-foreground"
          />
          <kbd className="rounded border px-1 py-px font-mono text-[10px] text-muted-foreground">
            esc
          </kbd>
        </div>
        <div className="overflow-y-auto py-2">
          {groups.length === 0 ? (
            <p className="px-5 py-6 text-center text-sm text-muted-foreground">
              No matches for &quot;{query}&quot;.
            </p>
          ) : (
            groups.map((g) => (
              <div key={g.title}>
                <p className="px-5 pb-1 pt-2.5 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                  {g.title}
                </p>
                {g.items.map((item) => {
                  const idx = flat.indexOf(item);
                  return (
                    <button
                      key={item.id}
                      type="button"
                      onClick={() => {
                        if (item.href) router.push(item.href);
                        else item.onSelect?.();
                        onClose();
                      }}
                      onMouseEnter={() => setActive(idx)}
                      className={cn(
                        "flex w-full items-center gap-3 px-5 py-2 text-left text-sm transition-colors",
                        idx === safeActive && "bg-accent text-accent-foreground",
                      )}
                    >
                      <item.Icon className="h-4 w-4 shrink-0 text-muted-foreground" aria-hidden />
                      <span className="flex-1">{item.label}</span>
                      <span className="text-xs text-muted-foreground">{item.meta}</span>
                    </button>
                  );
                })}
              </div>
            ))
          )}
        </div>
        <div className="flex gap-4 border-t bg-secondary px-5 py-2.5 text-[10.5px] text-muted-foreground">
          <span>
            <Kbd>↑</Kbd> <Kbd>↓</Kbd> navigate
          </span>
          <span>
            <Kbd>↵</Kbd> select
          </span>
          <span>
            <Kbd>esc</Kbd> close
          </span>
        </div>
      </div>
    </div>
  );
}

function Kbd({ children }: { children: React.ReactNode }) {
  return (
    <kbd className="rounded border bg-card px-1 py-px font-mono text-[9.5px] text-foreground">
      {children}
    </kbd>
  );
}
