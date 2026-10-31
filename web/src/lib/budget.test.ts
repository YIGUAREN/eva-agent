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

  it("defaults every extension to the exact value used by the form", () => {
    expect(defaultExtension(budget, usage)).toEqual({
      max_cost_usd: 1.9,
      max_wall_clock_s: 90,
      max_target_requests: 20,
    });
  });

  it("seeds the request default above aig_requests too", () => {
    const aigHeavy = { ...usage, target_requests: 4, aig_requests: 33 };
    const next = defaultExtension(budget, aigHeavy);
    expect(next.max_target_requests).toBe(43);
    expect(validateExtension(next, aigHeavy)).toBeNull();
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

  it("preserves the server message on an overrun", () => {
    expect(
      budgetDecisionError({
        status: 409,
        code: "budget_overrun",
        message: "finalize estimate exceeds the remaining budget",
        details: { finalize_estimate: 0.2, remaining: 0.1 },
      }),
    ).toEqual({
      kind: "overrun",
      finalizeEstimate: 0.2,
      remaining: 0.1,
      message: "finalize estimate exceeds the remaining budget",
    });
  });

  it("maps a rejected limit to the field name and server message", () => {
    expect(
      budgetDecisionError({
        status: 400,
        code: "invalid_request",
        message: "max_cost_usd must be greater than spent_usd",
        details: { field: "max_cost_usd" },
      }),
    ).toEqual({
      kind: "field",
      field: "max_cost_usd",
      message: "费用上限 must be greater than spent_usd",
    });
  });

  it("replaces the raw field key in the backend message with its human label", () => {
    const cases = [
      ["max_cost_usd", "费用上限", "新预算 max_cost_usd 必须严格大于已消费 0.9", "0.9"],
      ["max_wall_clock_s", "时长上限", "新预算 max_wall_clock_s 必须严格大于已消费 30.0", "30.0"],
      ["max_target_requests", "目标请求数上限", "新预算 max_target_requests 必须严格大于已消费 4", "4"],
    ] as const;

    for (const [field, label, serverMessage, number] of cases) {
      const failure = budgetDecisionError({
        status: 400,
        code: "invalid_request",
        message: serverMessage,
        details: { field },
      });
      expect(failure.kind).toBe("field");
      if (failure.kind !== "field") continue;
      expect(failure.message).toContain(label);
      expect(failure.message).toContain(number);
      expect(failure.message).not.toContain(field);
    }
  });

  it("falls back to a human field label when the server message is empty", () => {
    expect(
      budgetDecisionError({
        status: 400,
        code: "invalid_request",
        details: { field: "max_cost_usd" },
      }),
    ).toEqual({ kind: "field", field: "max_cost_usd", message: "费用上限 必须严格大于已消费值" });
  });

  it("preserves the server message for pending and state errors", () => {
    expect(
      budgetDecisionError({
        status: 409,
        code: "command_pending",
        message: "another command is already pending",
      }),
    ).toEqual({ kind: "message", message: "another command is already pending" });
    expect(
      budgetDecisionError({
        status: 409,
        code: "invalid_state",
        message: "run is not waiting for a budget decision",
      }),
    ).toEqual({ kind: "message", message: "run is not waiting for a budget decision" });
  });

  it("uses the server message for a FastAPI-style 422 payload", () => {
    expect(
      budgetDecisionError({
        status: 422,
        code: "invalid_request",
        message: "field required",
      }),
    ).toEqual({ kind: "message", message: "field required" });
  });

  it("falls back to the generic message when the server message is blank", () => {
    expect(
      budgetDecisionError({ status: 409, code: "invalid_state", message: "   " }),
    ).toEqual({ kind: "message", message: "预算决策提交失败" });
  });

  it("only treats budget_overrun as an overrun on a 409", () => {
    expect(
      budgetDecisionError({ status: 400, code: "budget_overrun", message: "bad request" }),
    ).toEqual({ kind: "message", message: "bad request" });
  });
});
