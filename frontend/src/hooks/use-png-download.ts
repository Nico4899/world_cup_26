"use client";

import { useCallback, useRef, useState } from "react";

/**
 * Snapshot a DOM subtree to PNG and trigger a browser download.
 *
 * Wraps `html-to-image` so chart cards can offer one-click PNG export
 * without each component owning the lifecycle. Replaces the Plotly modebar
 * download button that lived on every Streamlit chart.
 *
 * The dynamic import keeps the ~70 KB html-to-image bundle out of pages
 * that never call `download()`.
 */
export function usePngDownload(filename: string) {
  const ref = useRef<HTMLDivElement | null>(null);
  const [isDownloading, setIsDownloading] = useState(false);

  const download = useCallback(async () => {
    if (!ref.current) return;
    setIsDownloading(true);
    try {
      const { toPng } = await import("html-to-image");
      const dataUrl = await toPng(ref.current, {
        cacheBust: true,
        pixelRatio: 2,
        backgroundColor: "white",
      });
      const link = document.createElement("a");
      link.download = `${filename}.png`;
      link.href = dataUrl;
      link.click();
    } finally {
      setIsDownloading(false);
    }
  }, [filename]);

  return { ref, download, isDownloading };
}
