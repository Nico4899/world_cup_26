"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  CalendarClock,
  Trophy,
  LayoutGrid,
  GitBranch,
  LineChart,
  Info,
  Terminal,
  Users,
  MapPinned,
  type LucideIcon,
} from "lucide-react";

import { cn } from "@/lib/utils";

type NavItem = {
  href: string;
  label: string;
  Icon: LucideIcon;
};

const NAV: NavItem[] = [
  { href: "/", label: "Today", Icon: CalendarClock },
  { href: "/match/0", label: "Match Detail", Icon: Trophy },
  { href: "/groups", label: "Groups", Icon: LayoutGrid },
  { href: "/bracket", label: "Bracket", Icon: GitBranch },
  { href: "/track-record", label: "Track Record", Icon: LineChart },
  { href: "/about", label: "About", Icon: Info },
  { href: "/ops", label: "Operator", Icon: Terminal },
  { href: "/team/Argentina", label: "Team Profile", Icon: Users },
  { href: "/map", label: "Map", Icon: MapPinned },
];

function isActive(pathname: string, href: string) {
  if (href === "/") return pathname === "/";
  // Match `/match/...` from `/match/0`, `/team/...` from `/team/Argentina`, etc.
  const root = "/" + href.split("/")[1];
  return pathname === href || pathname.startsWith(root + "/");
}

export function Sidebar() {
  const pathname = usePathname();
  return (
    <nav className="flex flex-col gap-1 p-4">
      <div className="px-2 pb-3">
        <h1 className="text-base font-semibold tracking-tight">WC 2026</h1>
        <p className="text-xs text-muted-foreground">Calibrated predictions</p>
      </div>
      {NAV.map(({ href, label, Icon }) => {
        const active = isActive(pathname, href);
        return (
          <Link
            key={href}
            href={href}
            className={cn(
              "flex items-center gap-2 rounded-md px-3 py-2 text-sm transition-colors",
              "hover:bg-accent hover:text-accent-foreground",
              active && "bg-accent text-accent-foreground font-medium",
            )}
          >
            <Icon className="h-4 w-4" aria-hidden />
            <span>{label}</span>
          </Link>
        );
      })}
    </nav>
  );
}
