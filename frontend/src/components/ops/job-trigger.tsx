"use client";

import { Play } from "lucide-react";
import { useTransition, useState } from "react";

import { runJob } from "@/app/ops/actions";
import { Button } from "@/components/ui/button";

/** Single "Run {name}" button that calls the server action. */
export function JobTrigger({ name }: { name: string }) {
  const [isPending, startTransition] = useTransition();
  const [feedback, setFeedback] = useState<string | null>(null);

  return (
    <div className="space-y-1">
      <Button
        size="sm"
        variant="outline"
        disabled={isPending}
        onClick={() =>
          startTransition(async () => {
            const result = await runJob(name);
            setFeedback(
              result.ok
                ? `Enqueued (${result.status ?? "ok"})`
                : `Failed: ${result.error}`,
            );
          })
        }
      >
        <Play className="h-3.5 w-3.5 mr-1" aria-hidden />
        Run {name}
      </Button>
      {feedback ? (
        <p
          className={
            feedback.startsWith("Failed")
              ? "text-xs text-destructive"
              : "text-xs text-muted-foreground"
          }
        >
          {feedback}
        </p>
      ) : null}
    </div>
  );
}
