/**
 * Poster-stylised FIFA-cup brand mark for WC 2026. Mirrors the bundle's
 * BrandMark.jsx — globe, two handle-figures, tapered mint cup with a magenta
 * star punched in, stepped navy/green pedestal, three scallop dots on the
 * base. All flat fills; no gradients.
 *
 * Two variants:
 *   - "color" (default) — full poster treatment for chrome.
 *   - "solid"           — single-color via currentColor for icon-sized use.
 */

type Variant = "color" | "solid";

export function TrophyMark({
  size = 32,
  variant = "color",
}: {
  size?: number;
  variant?: Variant;
}) {
  const figLeft = "M 20 30 Q 22 22 32 22 L 32 36 Q 26 38 22 42 Q 18 38 20 30 Z";
  const figRight = "M 60 30 Q 58 22 48 22 L 48 36 Q 54 38 58 42 Q 62 38 60 30 Z";
  const cup =
    "M 22 42 Q 20 52 28 58 Q 32 60 32 64 L 48 64 Q 48 60 52 58 Q 60 52 58 42 Z";
  const star = "40,46 42.5,52 49,52.5 44,56.5 45.5,63 40,59.5 34.5,63 36,56.5 31,52.5 37.5,52";
  const h = Math.round(size * 1.25);

  if (variant === "color") {
    return (
      <svg
        width={size}
        height={h}
        viewBox="0 0 80 100"
        aria-label="WC 2026"
        role="img"
      >
        <circle cx="40" cy="14" r="9" fill="#e2245e" />
        <ellipse cx="40" cy="14" rx="9" ry="3.5" fill="none" stroke="#f3ead7" strokeWidth="1.5" />
        <line x1="40" y1="5" x2="40" y2="23" stroke="#f3ead7" strokeWidth="1.5" />
        <path d={figLeft} fill="#0a9b54" />
        <path d={figRight} fill="#0a9b54" />
        <path d={cup} fill="#9fdfc9" />
        <polygon points={star} fill="#e2245e" />
        <rect x="34" y="64" width="12" height="6" fill="#1d2188" />
        <rect x="28" y="70" width="24" height="6" rx="1.5" fill="#0a9b54" />
        <rect x="24" y="76" width="32" height="5" rx="1.5" fill="#1d2188" />
        <rect x="20" y="81" width="40" height="6" rx="2" fill="#0a9b54" />
        <circle cx="28" cy="84" r="1.2" fill="#f3ead7" />
        <circle cx="35" cy="84" r="1.2" fill="#f3ead7" />
        <circle cx="40" cy="84" r="1.2" fill="#e2245e" />
        <circle cx="45" cy="84" r="1.2" fill="#f3ead7" />
        <circle cx="52" cy="84" r="1.2" fill="#f3ead7" />
      </svg>
    );
  }

  return (
    <svg
      width={size}
      height={h}
      viewBox="0 0 80 100"
      aria-label="WC 2026"
      role="img"
    >
      <circle cx="40" cy="14" r="9" fill="currentColor" />
      <path d={figLeft} fill="currentColor" />
      <path d={figRight} fill="currentColor" />
      <path d={cup} fill="currentColor" />
      <polygon points={star} fill="var(--background)" />
      <rect x="34" y="64" width="12" height="6" fill="currentColor" />
      <rect x="28" y="70" width="24" height="6" rx="1.5" fill="currentColor" />
      <rect x="24" y="76" width="32" height="5" rx="1.5" fill="currentColor" />
      <rect x="20" y="81" width="40" height="6" rx="2" fill="currentColor" />
    </svg>
  );
}

export function BrandLockup({ size = 28 }: { size?: number }) {
  return (
    <div className="flex items-center gap-2.5">
      <TrophyMark size={size} variant="color" />
      <div className="flex flex-col leading-none whitespace-nowrap">
        <span className="text-base font-bold tracking-tight">WC 2026</span>
        <span className="text-[11px] text-muted-foreground mt-0.5">
          Calibrated predictions
        </span>
      </div>
    </div>
  );
}
