"use client";

import { useEffect, useState } from "react";

import { cn } from "@/lib/utils";

type Heading = { id: string; text: string; level: 1 | 2 | 3 };

/**
 * Sticky right-rail ToC for the About / Methodology page. Scans the
 * rendered MDX article for `<h1 id> <h2 id> <h3 id>` elements after mount
 * and tracks the active section with an IntersectionObserver.
 *
 * Collapses into a `<details>` block below the `lg` breakpoint so the
 * mobile reading order stays "intro → ToC → body."
 */
export function MethodologyToC({ articleSelector = "article" }: { articleSelector?: string }) {
  const [headings, setHeadings] = useState<Heading[]>([]);
  const [active, setActive] = useState<string | null>(null);

  useEffect(() => {
    const article = document.querySelector(articleSelector);
    if (!article) return;
    const nodes = Array.from(
      article.querySelectorAll<HTMLHeadingElement>("h1[id], h2[id], h3[id]"),
    );
    const collected: Heading[] = nodes.map((el) => ({
      id: el.id,
      text: el.textContent ?? "",
      level: Number(el.tagName[1]) as 1 | 2 | 3,
    }));
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setHeadings(collected);

    const obs = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
        if (visible[0]) setActive(visible[0].target.id);
      },
      { rootMargin: "-20% 0px -60% 0px" },
    );
    nodes.forEach((el) => obs.observe(el));
    return () => obs.disconnect();
  }, [articleSelector]);

  if (headings.length === 0) return null;

  const items = (
    <ul className="space-y-1 text-xs">
      {headings.map((h) => (
        <li key={h.id}>
          <a
            href={`#${h.id}`}
            className={cn(
              "block py-0.5 text-muted-foreground hover:text-foreground transition-colors",
              h.level === 2 && "pl-2",
              h.level === 3 && "pl-4",
              active === h.id && "text-foreground font-medium",
            )}
          >
            {h.text}
          </a>
        </li>
      ))}
    </ul>
  );

  return (
    <>
      <nav
        aria-label="On this page"
        className="hidden lg:block sticky top-20 max-h-[calc(100vh-6rem)] overflow-y-auto"
      >
        <p className="font-semibold text-muted-foreground uppercase tracking-wide text-[10px] mb-2">
          On this page
        </p>
        {items}
      </nav>
      <details className="lg:hidden rounded-md border bg-card p-3 mb-2">
        <summary className="cursor-pointer text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          On this page
        </summary>
        <div className="mt-2">{items}</div>
      </details>
    </>
  );
}
