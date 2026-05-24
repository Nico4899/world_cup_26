"use client";

import { Download } from "lucide-react";
import type { ReactNode } from "react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { usePngDownload } from "@/hooks/use-png-download";

type Props = {
  /** Card title. Accepts ReactNode so callers can append inline help affordances. */
  title: ReactNode;
  filename: string;
  children: ReactNode;
};

/**
 * Card with a header-level "Download PNG" button. Snapshots the card body
 * (excluding the header) so the exported image carries the chart + caption
 * without the button itself.
 */
export function DownloadableCard({ title, filename, children }: Props) {
  const { ref, download, isDownloading } = usePngDownload(filename);
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0">
        <CardTitle className="text-base">{title}</CardTitle>
        <Button
          size="sm"
          variant="ghost"
          onClick={download}
          disabled={isDownloading}
        >
          <Download className="h-3.5 w-3.5 mr-1" aria-hidden />
          PNG
        </Button>
      </CardHeader>
      <CardContent>
        <div ref={ref} className="bg-background p-2 rounded-md">
          {children}
        </div>
      </CardContent>
    </Card>
  );
}
