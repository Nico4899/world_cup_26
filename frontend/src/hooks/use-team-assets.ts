"use client";

import { useQuery } from "@tanstack/react-query";

import { ApiError, apiGet } from "@/lib/api";
import type { TeamAssetsResponse } from "@/lib/types";

const EMPTY = (team: string): TeamAssetsResponse => ({
  team,
  crest_url: null,
  kit_home_color: null,
  kit_away_color: null,
  stadium_name: null,
  stadium_capacity: null,
  stadium_city: null,
  stadium_country: null,
});

/**
 * 1-hour cached crest / kit / stadium lookup for one team.
 *
 * Falls back to an all-null payload on 404 / 503 so callers don't have to
 * branch on error state — same shape as the Streamlit team_assets helper.
 */
export function useTeamAssets(team: string) {
  return useQuery({
    queryKey: ["team-assets", team],
    queryFn: async () => {
      try {
        return await apiGet<TeamAssetsResponse>(`/api/v1/teams/${team}/assets`);
      } catch (err) {
        if (err instanceof ApiError && (err.status === 404 || err.status === 503)) {
          return EMPTY(team);
        }
        throw err;
      }
    },
    staleTime: 60 * 60 * 1000, // 1 hour, mirroring the Streamlit cache TTL
    placeholderData: EMPTY(team),
  });
}
