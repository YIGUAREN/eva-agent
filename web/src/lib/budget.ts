import type { Budget, BudgetGate, RunUsageView } from "../api/types";

export type BudgetDecisionFailure =
  | { kind: "overrun"; finalizeEstimate: number; remaining: number; message?: string }
  | { kind: "field"; field: string; message: string }
  | { kind: "message"; message: string };

const FIELD_LABELS: Record<string, string> = {
  max_cost_usd: "费用上限",
  max_wall_clock_s: "时长上限",
  max_target_requests: "目标请求数上限",
};

export function gateNeedsConfirmation(gate: BudgetGate): boolean {
  return gate.remaining < gate.finalize_estimate;
}

export function defaultExtension(budget: Budget, usage: RunUsageView): Budget {
  return {
    max_cost_usd: roundUp(Math.max(budget.max_cost_usd, usage.spent_usd + 1)),
    max_wall_clock_s: Math.ceil(Math.max(budget.max_wall_clock_s, usage.spent_wall_clock_s + 60)),
    max_target_requests:
      Math.max(budget.max_target_requests, usage.target_requests, usage.aig_requests) + 10,
  };
}

function roundUp(value: number): number {
  return Math.ceil(value * 1_000_000) / 1_000_000;
}

export function validateExtension(
  budget: Budget,
  usage: RunUsageView,
): { field: "max_cost_usd" | "max_wall_clock_s" | "max_target_requests"; message: string } | null {
  if (!(budget.max_cost_usd > usage.spent_usd)) {
    return { field: "max_cost_usd", message: `成本上限必须大于已消费 ${usage.spent_usd}` };
  }
  if (!(budget.max_wall_clock_s > usage.spent_wall_clock_s)) {
    return { field: "max_wall_clock_s", message: `时长上限必须大于已消费 ${usage.spent_wall_clock_s} 秒` };
  }
  const requests = Math.max(usage.target_requests, usage.aig_requests);
  if (!(budget.max_target_requests > requests)) {
    return { field: "max_target_requests", message: `请求上限必须大于已消费 ${requests}` };
  }
  return null;
}

export function budgetDecisionError(error: {
  status: number;
  code: string;
  message?: string;
  details?: Record<string, unknown>;
}): BudgetDecisionFailure {
  const message = error.message?.trim() ?? "";
  if (error.status === 409 && error.code === "budget_overrun") {
    return {
      kind: "overrun",
      finalizeEstimate: Number(error.details?.finalize_estimate ?? 0),
      remaining: Number(error.details?.remaining ?? 0),
      ...(message ? { message } : {}),
    };
  }
  if (error.code === "invalid_request" && typeof error.details?.field === "string") {
    const field = error.details.field;
    const label = FIELD_LABELS[field] ?? field;
    return {
      kind: "field",
      field,
      message:
        message && field
          ? message.split(field).join(label)
          : message || `${label} 必须严格大于已消费值`,
    };
  }
  return { kind: "message", message: message || "预算决策提交失败" };
}
