import { describe, expect, it } from "vitest";

import {
  budgetDecisionError,
  defaultExtension,
  gateNeedsConfirmation,
  validateExtension,
} from "./budget";

const budget = { max_cost_usd: 1, max_wall_clock_s: 60, max_target_requests: 10 };
const usage = {
  spent_usd: 0.9,
  reserved_usd: 0,
  spent_wall_clock_s: 30,
  target_requests: 4,
  aig_requests: 0,
};

describe("budget helpers", () => {
  it("asks for confirmation only when remaining is below the finalize estimate", () => {
    expect(
      gateNeedsConfirmation({
        finalize_estimate: 0.2,
        remaining: 0.3,
        spent: 0.7,
        truncated_dimensions: ["quality"],
      }),
    ).toBe(false);
    expect(
      gateNeedsConfirmation({
        finalize_estimate: 0.2,
        remaining: 0.1,
        spent: 0.9,
        truncated_dimensions: ["quality"],
      }),
    ).toBe(true);
  });

  it("defaults every extension strictly above what was consumed", () => {
    const next = defaultExtension(budget, usage);
    expect(next.max_cost_usd).toBeGreaterThan(usage.spent_usd);
    expect(next.max_wall_clock_s).toBeGreaterThan(usage.spent_wall_clock_s);
    expect(next.max_target_requests).toBeGreaterThan(usage.target_requests);
  });

  it("reports the first limit that is not strictly larger", () => {
    expect(
      validateExtension({ ...budget, max_cost_usd: 0.9 }, usage)?.field,
    ).toBe("max_cost_usd");
    expect(
      validateExtension({ ...budget, max_wall_clock_s: 30 }, usage)?.field,
    ).toBe("max_wall_clock_s");
    expect(
      validateExtension({ ...budget, max_target_requests: 4 }, usage)?.field,
    ).toBe("max_target_requests");
    expect(validateExtension(budget, usage)).toBeNull();
  });

  it("maps an overrun envelope to a confirmation prompt", () => {
    expect(
      budgetDecisionError({
        status: 409,
        code: "budget_overrun",
        details: { finalize_estimate: 0.2, remaining: 0.1 },
      }),
    ).toEqual({ kind: "overrun", finalizeEstimate: 0.2, remaining: 0.1 });
  });

  it("maps a rejected limit to the field name", () => {
    expect(
      budgetDecisionError({
        status: 400,
        code: "invalid_request",
        details: { field: "max_cost_usd" },
      }),
    ).toEqual({ kind: "field", field: "max_cost_usd" });
  });
});
