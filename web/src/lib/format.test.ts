import { describe, expect, it } from "vitest";

import { spend, yuan } from "./format";

describe("yuan", () => {
  it("prefixes the symbol and keeps two decimals by default", () => {
    expect(yuan(1.23456)).toBe("¥1.23");
  });
  it("honours an explicit precision for small budget amounts", () => {
    expect(yuan(0.0001234, 6)).toBe("¥0.000123");
  });
  it("returns a dash for a non-finite value", () => {
    expect(yuan(Number.NaN)).toBe("—");
    expect(yuan(Number.POSITIVE_INFINITY)).toBe("—");
  });
});

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
