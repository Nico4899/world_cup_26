import { describe, it, expect, beforeEach } from "vitest";
import { renderHook, act } from "@testing-library/react";

import { useLockedBracket } from "./use-locked-bracket";

describe("useLockedBracket", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  it("starts empty", () => {
    const { result } = renderHook(() => useLockedBracket());
    expect(result.current.locks).toEqual([]);
  });

  it("add() appends a lock", () => {
    const { result } = renderHook(() => useLockedBracket());
    act(() => result.current.add({ match_id: 73, winner: "Argentina" }));
    expect(result.current.locks).toEqual([{ match_id: 73, winner: "Argentina" }]);
  });

  it("add() overwrites an existing lock for the same match_id", () => {
    const { result } = renderHook(() => useLockedBracket());
    act(() => result.current.add({ match_id: 73, winner: "Argentina" }));
    act(() => result.current.add({ match_id: 73, winner: "Spain" }));
    expect(result.current.locks).toEqual([{ match_id: 73, winner: "Spain" }]);
  });

  it("remove() drops the named match_id", () => {
    const { result } = renderHook(() => useLockedBracket());
    act(() => result.current.add({ match_id: 73, winner: "Argentina" }));
    act(() => result.current.add({ match_id: 89, winner: "Brazil" }));
    act(() => result.current.remove(73));
    expect(result.current.locks).toEqual([{ match_id: 89, winner: "Brazil" }]);
  });

  it("clear() empties the list", () => {
    const { result } = renderHook(() => useLockedBracket());
    act(() => result.current.add({ match_id: 73, winner: "Argentina" }));
    act(() => result.current.clear());
    expect(result.current.locks).toEqual([]);
  });

  it("persists across hook instances via localStorage", () => {
    const { result, unmount } = renderHook(() => useLockedBracket());
    act(() => result.current.add({ match_id: 104, winner: "France" }));
    unmount();

    const { result: result2 } = renderHook(() => useLockedBracket());
    expect(result2.current.locks).toEqual([{ match_id: 104, winner: "France" }]);
  });

  it("ignores malformed JSON in storage", () => {
    window.localStorage.setItem("wc2026.bracket_locks", "{not json");
    const { result } = renderHook(() => useLockedBracket());
    expect(result.current.locks).toEqual([]);
  });
});
