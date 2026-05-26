"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import {
  CalendarClock,
  Command,
  GitBranch,
  Info,
  LayoutGrid,
  LineChart,
  MapPinned,
  Menu,
  Terminal,
  Trophy,
  type LucideIcon,
} from "lucide-react";

import { BrandLockup } from "@/components/brand-mark";
import { CommandPalette } from "@/components/command-palette";
import { ThemeToggle } from "@/components/theme-toggle";
import { Sheet, SheetContent, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { cn } from "@/lib/utils";

type Tab = { href: string; label: string; Icon: LucideIcon };

const TABS: Tab[] = [
  { href: "/", label: "Today", Icon: CalendarClock },
  { href: "/match/0", label: "Match Detail", Icon: Trophy },
  { href: "/groups", label: "Groups", Icon: LayoutGrid },
  { href: "/bracket", label: "Bracket", Icon: GitBranch },
  { href: "/track-record", label: "Track Record", Icon: LineChart },
  { href: "/map", label: "Map", Icon: MapPinned },
  { href: "/about", label: "About", Icon: Info },
];

function isActive(pathname: string, href: string) {
  if (href === "/") return pathname === "/";
  const root = "/" + href.split("/")[1];
  return pathname === href || pathname.startsWith(root + "/");
}

export function TopNav() {
  const pathname = usePathname();
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [sheetOpen, setSheetOpen] = useState(false);

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

  // Close the mobile sheet automatically when the route changes.
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setSheetOpen(false);
  }, [pathname]);

  return (
    <>
      <header className="sticky top-0 z-10 flex h-14 items-center gap-3 border-b bg-card px-4">
        {/* Mobile hamburger — collapses the 5 primary tabs into a Sheet. */}
        <button
          type="button"
          onClick={() => setSheetOpen(true)}
          className="md:hidden inline-flex items-center justify-center rounded-md p-1.5 text-foreground transition-colors hover:bg-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          aria-label="Open primary navigation"
          aria-controls="primary-nav-sheet"
          aria-expanded={sheetOpen}
        >
          <Menu className="h-5 w-5" aria-hidden />
        </button>

        <Link
          href="/"
          className="shrink-0 md:border-r md:border-border md:pr-4"
          aria-label="WC 2026 home"
        >
          <BrandLockup size={28} />
        </Link>

        {/* Desktop inline tabs — hidden below the md breakpoint. */}
        <nav
          className="hidden md:flex flex-1 min-w-0 gap-0.5 overflow-hidden"
          aria-label="Primary"
        >
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
                    "after:absolute after:-bottom-3.25 after:left-3 after:right-3 after:h-0.75 after:rounded-sm after:bg-primary",
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

        {/* Spacer pushes the right-side actions to the edge on mobile too. */}
        <div className="flex-1 md:hidden" />

        <button
          type="button"
          onClick={() => setPaletteOpen(true)}
          className="inline-flex items-center gap-2 rounded-md border border-border bg-card px-2.5 py-1.5 text-xs text-muted-foreground transition-colors hover:bg-accent"
          aria-label="Open command palette"
          aria-expanded={paletteOpen}
          aria-controls="command-palette"
        >
          <Command className="h-3.5 w-3.5 md:hidden" aria-hidden />
          <span className="hidden md:inline">Jump to…</span>
          <kbd className="rounded border bg-accent px-1 py-px font-mono text-[10px] text-foreground">
            ⌘K
          </kbd>
        </button>
        {/* Operator stays outside the primary tabs (intentionally technical)
            but is one click away as an icon in the chrome. */}
        <Link
          href="/ops"
          aria-label="Operator"
          title="Operator"
          className={cn(
            "hidden md:inline-flex items-center justify-center rounded-md p-1.5 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
            isActive(pathname, "/ops") && "bg-accent text-foreground",
          )}
        >
          <Terminal className="h-4 w-4" aria-hidden />
        </Link>
        <ThemeToggle />
      </header>

      <Sheet open={sheetOpen} onOpenChange={setSheetOpen}>
        <SheetContent side="left" id="primary-nav-sheet" className="p-0 flex flex-col gap-0">
          <SheetHeader className="border-b">
            <SheetTitle>Navigation</SheetTitle>
          </SheetHeader>
          <nav className="flex flex-col p-2 gap-0.5" aria-label="Primary mobile">
            {TABS.map(({ href, label, Icon }) => {
              const active = isActive(pathname, href);
              return (
                <Link
                  key={href}
                  href={href}
                  className={cn(
                    "inline-flex items-center gap-3 rounded-md px-3 py-2 text-sm transition-colors",
                    "hover:bg-accent",
                    active && "bg-accent font-semibold",
                  )}
                >
                  <Icon
                    className="h-4 w-4"
                    aria-hidden
                    style={{ color: active ? "var(--foreground)" : "var(--muted-foreground)" }}
                  />
                  {label}
                </Link>
              );
            })}
          </nav>
          <div className="border-t p-2 flex flex-col gap-0.5">
            <Link
              href="/ops"
              className={cn(
                "inline-flex items-center gap-3 rounded-md px-3 py-2 text-sm transition-colors hover:bg-accent",
                isActive(pathname, "/ops") && "bg-accent font-semibold",
              )}
            >
              <Terminal
                className="h-4 w-4"
                aria-hidden
                style={{
                  color: isActive(pathname, "/ops")
                    ? "var(--foreground)"
                    : "var(--muted-foreground)",
                }}
              />
              Operator
              <span className="ml-auto text-[10px] uppercase tracking-wider text-muted-foreground">
                tools
              </span>
            </Link>
            <button
              type="button"
              onClick={() => {
                setSheetOpen(false);
                setPaletteOpen(true);
              }}
              className="w-full inline-flex items-center gap-3 rounded-md px-3 py-2 text-sm text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
            >
              <Command className="h-4 w-4" aria-hidden />
              Jump to anything…
              <kbd className="ml-auto rounded border bg-accent px-1 py-px font-mono text-[10px] text-foreground">
                ⌘K
              </kbd>
            </button>
          </div>
        </SheetContent>
      </Sheet>

      <CommandPalette open={paletteOpen} onClose={() => setPaletteOpen(false)} />
    </>
  );
}
