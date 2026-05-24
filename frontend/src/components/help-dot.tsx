"use client";

import Link from "next/link";
import { HelpCircle } from "lucide-react";

import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { GLOSSARY, type GlossaryKey } from "@/lib/glossary";
import { cn } from "@/lib/utils";

type CommonProps = {
  term: GlossaryKey;
  className?: string;
};

/**
 * Inline question-mark affordance that opens a popover with a one-sentence
 * plain-language gloss for a technical term. Click-to-open (not hover-only)
 * so it works for keyboard + touch users too.
 */
export function HelpDot({ term, className }: CommonProps) {
  const entry = GLOSSARY[term];
  if (!entry) return null;
  return (
    <Popover>
      <PopoverTrigger
        type="button"
        aria-label={`What is ${entry.name}?`}
        className={cn(
          "inline-flex items-center align-baseline ml-0.5 rounded-full text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
          className,
        )}
      >
        <HelpCircle className="h-3 w-3" aria-hidden />
      </PopoverTrigger>
      <PopoverContent className="w-72 space-y-2" side="top">
        <p className="text-sm font-medium">{entry.name}</p>
        <p className="text-xs text-muted-foreground">{entry.short}</p>
        {entry.link ? (
          <Link
            href={entry.link}
            className="inline-block text-xs underline underline-offset-2"
          >
            Read more in About
          </Link>
        ) : null}
      </PopoverContent>
    </Popover>
  );
}

/**
 * Wraps inline text with a subtle dotted underline that triggers the same
 * popover as <HelpDot/>. Use when the term itself should remain readable
 * as prose (e.g. "Expected goals" rather than "Expected goals ⓘ").
 */
export function TermHelp({
  term,
  children,
  className,
}: CommonProps & { children: React.ReactNode }) {
  const entry = GLOSSARY[term];
  if (!entry) return <>{children}</>;
  return (
    <Popover>
      <PopoverTrigger
        type="button"
        aria-label={`What is ${entry.name}?`}
        className={cn(
          "rounded-sm underline decoration-dotted decoration-from-font underline-offset-2 transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
          className,
        )}
      >
        {children}
      </PopoverTrigger>
      <PopoverContent className="w-72 space-y-2" side="top">
        <p className="text-sm font-medium">{entry.name}</p>
        <p className="text-xs text-muted-foreground">{entry.short}</p>
        {entry.link ? (
          <Link
            href={entry.link}
            className="inline-block text-xs underline underline-offset-2"
          >
            Read more in About
          </Link>
        ) : null}
      </PopoverContent>
    </Popover>
  );
}
