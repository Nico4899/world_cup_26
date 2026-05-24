/**
 * Slugify a heading string for use as an anchor `id`.
 *
 * Strips a leading outline prefix ("1. ", "3.1 ", "0. ") so anchor IDs
 * stay stable when sections get renumbered. That lets glossary.ts entries
 * point to `/about#match-model` rather than `/about#1-match-model`.
 *
 * Lower-cases, replaces every non-alphanumeric run with a single hyphen,
 * then trims hyphens from the ends.
 */
export function slugify(text: string): string {
  return text
    .toLowerCase()
    .replace(/^\s*\d+(\.\d+)*\.?\s+/, "")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}
