import { describe, it, expect } from "vitest";

import { utcTimeOfDay, pct, signed, prettyDate } from "./format";

describe("utcTimeOfDay", () => {
  it("slices HH:MM out of an ISO timestamp", () => {
    expect(utcTimeOfDay("2026-06-11T18:30:00+00:00")).toBe("18:30 UTC");
  });

  it("returns null for malformed input", () => {
    expect(utcTimeOfDay(null)).toBeNull();
    expect(utcTimeOfDay(undefined)).toBeNull();
    expect(utcTimeOfDay("")).toBeNull();
    expect(utcTimeOfDay("2026-06-11")).toBeNull(); // too short, no T
    expect(utcTimeOfDay("2026-06-11 18:30 UTC")).toBeNull(); // T missing at index 10
  });
});

describe("pct", () => {
  it("formats with default 1 decimal", () => {
    expect(pct(0.482)).toBe("48.2%");
  });

  it("respects custom digits", () => {
    expect(pct(0.482, 0)).toBe("48%");
    expect(pct(0.482, 3)).toBe("48.200%");
  });
});

describe("signed", () => {
  it("prefixes positive values with +", () => {
    expect(signed(0.42)).toBe("+0.42");
  });

  it("preserves the minus on negatives", () => {
    expect(signed(-0.42)).toBe("-0.42");
  });

  it("treats zero as positive", () => {
    expect(signed(0)).toBe("+0.00");
  });
});

describe("prettyDate", () => {
  it("returns a weekday + month abbreviation", () => {
    const out = prettyDate("2026-06-11");
    expect(out).toContain("Jun");
    expect(out).toContain("11");
    expect(out).toContain("2026");
  });

  it("returns the input on parse failure", () => {
    expect(prettyDate("not a date")).toBe("not a date");
  });
});
