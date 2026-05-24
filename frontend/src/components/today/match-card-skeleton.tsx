import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";

/**
 * Placeholder card that mirrors the rough geometry of <MatchCard/> so the
 * Today grid doesn't reflow when its async siblings resolve. Used as the
 * <Suspense fallback> while each card streams its own prediction fetch.
 */
export function MatchCardSkeleton() {
  return (
    <Card variant="ribbon">
      <CardContent className="space-y-3 py-4">
        <div className="flex items-center justify-between gap-2">
          <Skeleton className="h-6 w-32" />
          <Skeleton className="h-6 w-20" />
        </div>
        <Skeleton className="h-3 w-48" />
        <Skeleton className="h-8 w-full" />
        <Skeleton className="h-3 w-44" />
        <div className="flex items-center justify-between pt-1">
          <div className="space-y-1">
            <Skeleton className="h-3 w-24" />
            <Skeleton className="h-4 w-16" />
          </div>
          <div className="space-y-1 items-end flex flex-col">
            <Skeleton className="h-3 w-20" />
            <Skeleton className="h-3 w-16" />
            <Skeleton className="h-3 w-16" />
          </div>
        </div>
        <div className="grid grid-cols-2 gap-2 pt-1">
          <Skeleton className="h-8 w-full" />
          <Skeleton className="h-8 w-full" />
        </div>
      </CardContent>
    </Card>
  );
}
