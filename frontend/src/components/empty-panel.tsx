import Link from "next/link";
import { Inbox, type LucideIcon } from "lucide-react";

import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";

type Props = {
  /** Lucide icon. Defaults to <Inbox/> when omitted. */
  Icon?: LucideIcon;
  /** One-line heading, sentence case. */
  title: string;
  /** Optional explanatory copy below the heading. */
  hint?: React.ReactNode;
  /** Optional call-to-action — a single link to an alternative surface. */
  cta?: { href: string; label: string };
  className?: string;
};

/**
 * Friendly empty-state panel. Use anywhere a section would otherwise render
 * a terse "No data" italic line. The panel keeps card padding consistent so
 * a missing section doesn't shrink the surrounding layout.
 */
export function EmptyPanel({ Icon = Inbox, title, hint, cta, className }: Props) {
  return (
    <Card className={cn("border-dashed", className)}>
      <CardContent className="py-6 text-center space-y-2">
        <Icon
          className="mx-auto h-6 w-6 text-muted-foreground"
          aria-hidden
          strokeWidth={1.6}
        />
        <p className="text-sm font-medium">{title}</p>
        {hint ? (
          <p className="text-xs text-muted-foreground max-w-md mx-auto">{hint}</p>
        ) : null}
        {cta ? (
          <Link
            href={cta.href}
            className="inline-block text-xs underline underline-offset-2 hover:text-foreground"
          >
            {cta.label}
          </Link>
        ) : null}
      </CardContent>
    </Card>
  );
}
