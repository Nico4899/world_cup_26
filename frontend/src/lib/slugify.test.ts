import { readFileSync } from "node:fs";
import { join } from "node:path";

import { describe, it, expect } from "vitest";

import { GLOSSARY } from "./glossary";
import { slugify } from "./slugify";

describe("slugify()", () => {
  it("lower-cases, replaces non-alphanumeric runs with hyphens, and trims", () => {
    expect(slugify("Hello World")).toBe("hello-world");
    expect(slugify("  spaces  on the edges  ")).toBe("spaces-on-the-edges");
    expect(slugify("Already-Slugged_text!")).toBe("already-slugged-text");
  });

  it("strips a leading outline prefix ('1. ', '3.1 ', '0. ')", () => {
    expect(slugify("0. Lineage")).toBe("lineage");
    expect(slugify("1. Match model")).toBe("match-model");
    expect(slugify("3.1 Elo prior on attack/defence")).toBe(
      "elo-prior-on-attack-defence",
    );
    expect(slugify("4. Tournament simulator + tiebreakers")).toBe(
      "tournament-simulator-tiebreakers",
    );
    expect(slugify("5. Backtest gates")).toBe("backtest-gates");
    expect(slugify("3. Stage 1 enhancements (this PR)")).toBe(
      "stage-1-enhancements-this-pr",
    );
  });

  it("leaves headings without an outline prefix alone (apart from slugification)", () => {
    expect(slugify("Methodology")).toBe("methodology");
    expect(slugify("Limitations / open issues")).toBe("limitations-open-issues");
  });

  it("returns an empty string for empty / whitespace-only input", () => {
    expect(slugify("")).toBe("");
    expect(slugify("   ")).toBe("");
  });
});

/**
 * Invariant test: every `glossary.link?: "/about#…"` must point to a
 * heading that actually exists in the methodology doc. If a heading is
 * renamed without updating the glossary, this test fails — preventing
 * silent broken popover deep links.
 */
describe("glossary <-> methodology cross-check", () => {
  const docPath = join(
    process.cwd(),
    "..",
    "docs",
    "methodology.md",
  );
  const methodology = readFileSync(docPath, "utf8");
  const headingSlugs = new Set(
    methodology
      .split("\n")
      .filter((line) => /^#{1,3}\s+/.test(line))
      .map((line) => line.replace(/^#{1,3}\s+/, ""))
      .map(slugify),
  );

  const linkedTerms = Object.entries(GLOSSARY).filter(([, entry]) =>
    Boolean(entry.link),
  );

  it.each(linkedTerms)("%s links to a known methodology heading", (_, entry) => {
    const slug = entry.link!.replace(/^\/about#/, "");
    expect(headingSlugs).toContain(slug);
  });
});
