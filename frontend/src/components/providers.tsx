"use client";

import type { ReactNode } from "react";
import { ThemeProvider } from "next-themes";

import { TooltipProvider } from "@/components/ui/tooltip";
import { ReactQueryProvider } from "@/lib/query-client";

/**
 * Composition root for every client-side provider.
 *
 * Order matters: ReactQueryProvider is innermost so a Tooltip's hook can read
 * from the cache during render; ThemeProvider is outermost so children see the
 * resolved class on first paint.
 */
export function Providers({ children }: { children: ReactNode }) {
  return (
    <ThemeProvider
      attribute="class"
      defaultTheme="system"
      enableSystem
      disableTransitionOnChange
    >
      <ReactQueryProvider>
        <TooltipProvider delay={150}>{children}</TooltipProvider>
      </ReactQueryProvider>
    </ThemeProvider>
  );
}
