"use client";

import { useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import {
  CalendarDays,
  Command,
  LineChart,
  MousePointerClick,
  X,
  type LucideIcon,
} from "lucide-react";

import { Button } from "@/components/ui/button";

const STORAGE_KEY = "wc2026-onboarded";

type Step = {
  Icon: LucideIcon;
  title: string;
  body: string;
};

const STEPS: Step[] = [
  {
    Icon: CalendarDays,
    title: "Each card is one match",
    body: "Every fixture scheduled today shows the model's odds for home win / draw / away win, the three most-likely scorelines, and a one-tap explainer.",
  },
  {
    Icon: MousePointerClick,
    title: "Every probability is clickable",
    body: "Outcome tiles, group bars, and metric numbers open a popover with the inputs driving the prediction. Anything with a ⓘ has a one-sentence definition.",
  },
  {
    Icon: Command,
    title: "Jump anywhere with ⌘K",
    body: "Press Cmd / Ctrl + K (or tap “Jump to…” in the top bar) to open the command palette and search routes, teams, and actions.",
  },
  {
    Icon: LineChart,
    title: "Check our work",
    body: "Track Record shows how close past predictions have been to reality. Lower log-loss is better; the reliability diagram tells you if the model is honest about its uncertainty.",
  },
];

/**
 * One-time onboarding overlay shown on the Today page on first visit.
 * State is local to the browser via localStorage (`wc2026-onboarded`).
 *
 * Visit /?reset-tour to clear the flag for demos.
 */
export function FirstVisitTour() {
  const router = useRouter();
  const params = useSearchParams();
  const [open, setOpen] = useState(false);
  const [step, setStep] = useState(0);

  // Hydrate from localStorage in an effect to avoid SSR/CSR text mismatch.
  useEffect(() => {
    const reset = params.get("reset-tour");
    if (reset !== null) {
      try {
        window.localStorage.removeItem(STORAGE_KEY);
      } catch {
        // Storage may be blocked (Safari private mode); just open the tour.
      }
      const next = new URLSearchParams(params.toString());
      next.delete("reset-tour");
      router.replace(next.toString() ? `/?${next.toString()}` : "/");
    }
    let seen = false;
    try {
      seen = window.localStorage.getItem(STORAGE_KEY) === "1";
    } catch {
      // Treat storage failures as "show once per page-load," which is the
      // graceful fallback for private-browsing visitors.
    }
    if (!seen) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setOpen(true);
    }
  }, [params, router]);

  function dismiss() {
    try {
      window.localStorage.setItem(STORAGE_KEY, "1");
    } catch {
      // ignore
    }
    setOpen(false);
  }

  if (!open) return null;
  const current = STEPS[step];
  const lastStep = step === STEPS.length - 1;
  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="tour-title"
      className="fixed inset-0 z-50 flex items-center justify-center bg-primary/30 backdrop-blur-sm px-4"
    >
      <div className="relative w-full max-w-md rounded-xl bg-card text-card-foreground p-6 shadow-xl ring-1 ring-foreground/10 space-y-4">
        <button
          type="button"
          onClick={dismiss}
          aria-label="Close walkthrough"
          className="absolute top-3 right-3 rounded-md p-1 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          <X className="h-4 w-4" aria-hidden />
        </button>
        <div className="flex items-start gap-3">
          <div className="rounded-full bg-secondary p-2.5 shrink-0">
            <current.Icon className="h-5 w-5" aria-hidden strokeWidth={2} />
          </div>
          <div className="space-y-1">
            <p className="text-[11px] uppercase tracking-wider text-muted-foreground">
              {step + 1} / {STEPS.length}
            </p>
            <h2 id="tour-title" className="text-lg font-semibold">
              {current.title}
            </h2>
          </div>
        </div>
        <p className="text-sm text-muted-foreground leading-6">
          {current.body}
        </p>
        <div className="flex items-center justify-between pt-2">
          <Button variant="ghost" size="sm" onClick={dismiss}>
            Skip tour
          </Button>
          <div className="flex gap-2">
            {step > 0 ? (
              <Button
                variant="outline"
                size="sm"
                onClick={() => setStep((s) => Math.max(0, s - 1))}
              >
                Back
              </Button>
            ) : null}
            <Button
              size="sm"
              onClick={() =>
                lastStep ? dismiss() : setStep((s) => Math.min(STEPS.length - 1, s + 1))
              }
            >
              {lastStep ? "Done" : "Next"}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
