"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import {
  CalendarClock,
  GitBranch,
  LayoutGrid,
  LineChart,
  Trophy,
  type LucideIcon,
} from "lucide-react";

import { BrandLockup } from "@/components/brand-mark";
import { CommandPalette } from "@/components/command-palette";
import { ThemeToggle } from "@/components/theme-toggle";
import { cn } from "@/lib/utils";

type Tab = { href: string; label: string; Icon: LucideIcon };

const TABS: Tab[] = [
  { href: "/", label: "Today", Icon: CalendarClock },
  { href: "/match/0", label: "Match Detail", Icon: Trophy },
  { href: "/groups", label: "Groups", Icon: LayoutGrid },
  { href: "/bracket", label: "Bracket", Icon: GitBranch },
  { href: "/track-record", label: "Track Record", Icon: LineChart },
];

function isActive(pathname: string, href: string) {
  if (href === "/") return pathname === "/";
  const root = "/" + href.split("/")[1];
  return pathname === href || pathname.startsWith(root + "/");
}

export function TopNav() {
  const pathname = usePathname();
  const [paletteOpen, setPaletteOpen] = useState(false);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setPaletteOpen((o) => !o);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  return (
    <>
      <header className="sticky top-0 z-10 flex h-14 items-center gap-3 border-b bg-card px-4">
        <Link
          href="/"
          className="flex-shrink-0 border-r border-border pr-4"
          aria-label="WC 2026 home"
        >
          <BrandLockup size={28} />
        </Link>
        <nav className="flex flex-1 min-w-0 gap-0.5 overflow-hidden" aria-label="Primary">
          {TABS.map(({ href, label, Icon }) => {
            const active = isActive(pathname, href);
            return (
              <Link
                key={href}
                href={href}
                className={cn(
                  "relative inline-flex items-center gap-1.5 whitespace-nowrap rounded-md px-3 py-1.5 text-sm transition-colors",
                  "hover:bg-accent hover:text-accent-foreground",
                  active && "bg-accent font-semibold text-accent-foreground",
                  active &&
                    "after:absolute after:-bottom-[13px] after:left-3 after:right-3 after:h-[3px] after:rounded-sm after:bg-primary",
                )}
              >
                <Icon
                  className="h-3.5 w-3.5"
                  aria-hidden
                  style={{ color: active ? "var(--foreground)" : "var(--muted-foreground)" }}
                />
                {label}
              </Link>
            );
          })}
        </nav>
        <button
          type="button"
          onClick={() => setPaletteOpen(true)}
          className="inline-flex items-center gap-2 rounded-md border border-border bg-card px-3 py-1.5 text-xs text-muted-foreground transition-colors hover:bg-accent"
          aria-label="Open command palette"
        >
          <span>Jump to…</span>
          <kbd className="rounded border bg-accent px-1 py-px font-mono text-[10px] text-foreground">
            ⌘K
          </kbd>
        </button>
        <ThemeToggle />
      </header>
      <CommandPalette open={paletteOpen} onClose={() => setPaletteOpen(false)} />
    </>
  );
}
