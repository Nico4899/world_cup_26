"use client";

import { AlertTriangle, RefreshCw } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { API_URL } from "@/lib/api";

type Props = {
  message?: string;
  onRetry?: () => void;
};

/**
 * Mirrors the Streamlit `render_unreachable_warning` helper: a single visible
 * banner that names the API URL the caller failed to reach. Used by every
 * page that fetches client-side data; server components surface a similar
 * message from their own catch blocks.
 */
export function ApiUnreachableBanner({ message, onRetry }: Props) {
  return (
    <Card className="border-destructive/40 bg-destructive/5">
      <CardContent className="flex flex-col gap-3 py-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-start gap-3">
          <AlertTriangle className="h-5 w-5 text-destructive mt-0.5" aria-hidden />
          <div className="space-y-1">
            <p className="text-sm font-medium">
              Couldn&apos;t reach the prediction API at{" "}
              <code className="rounded bg-muted px-1 py-0.5 text-xs">{API_URL}</code>.
            </p>
            <p className="text-xs text-muted-foreground">
              {message ??
                "Start it locally with `uv run uvicorn wc2026.api.main:app` or check the deployed Fly app."}
            </p>
          </div>
        </div>
        {onRetry ? (
          <Button size="sm" variant="outline" onClick={onRetry}>
            <RefreshCw className="h-3.5 w-3.5 mr-1.5" aria-hidden />
            Retry
          </Button>
        ) : null}
      </CardContent>
    </Card>
  );
}
