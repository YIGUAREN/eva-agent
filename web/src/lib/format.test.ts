import { describe, expect, it } from "vitest";

import { spend } from "./format";

describe("spend", () => {
  it("formats a fractional amount with two decimals", () => {
    expect(spend(12.345678)).toBe("¥12.35");
  });
  it("returns null for a missing value", () => {
    expect(spend(undefined)).toBeNull();
    expect(spend(null)).toBeNull();
  });
  it("returns null for a non-finite value", () => {
    expect(spend(Number.NaN)).toBeNull();
    expect(spend(Number.POSITIVE_INFINITY)).toBeNull();
  });
});
