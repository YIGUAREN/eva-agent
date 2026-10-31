import { useState } from "react";

import { ApiError, api } from "../api/client";
import type { Budget, BudgetGate, RunUsageView } from "../api/types";
import {
  budgetDecisionError,
  defaultExtension,
  gateNeedsConfirmation,
  validateExtension,
  type BudgetDecisionFailure,
} from "../lib/budget";
import { yuan } from "../lib/format";

export function BudgetCard({
  runId,
  gate,
  budget,
  usage,
  onDone,
}: {
  runId: string;
  gate: BudgetGate;
  budget: Budget;
  usage: RunUsageView;
  onDone: () => void;
}) {
  const [extension, setExtension] = useState<Budget>(() => defaultExtension(budget, usage));
  const [failure, setFailure] = useState<BudgetDecisionFailure | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(body: Parameters<typeof api.budgetDecision>[1]) {
    setBusy(true);
    setFailure(null);
    try {
      await api.budgetDecision(runId, body);
      onDone();
    } catch (error) {
      if (error instanceof ApiError) {
        setFailure(budgetDecisionError(error));
      } else {
        setFailure({ kind: "message", message: "网络错误，请重试" });
      }
    } finally {
      setBusy(false);
    }
  }

  function extend() {
    const invalid = validateExtension(extension, usage);
    if (invalid) {
      setFailure({ kind: "field", field: invalid.field, message: invalid.message });
      return;
    }
    void submit({ action: "extend", budget: extension });
  }

  const needsConfirmation = failure?.kind === "overrun" || gateNeedsConfirmation(gate);

  return (
    <section
      aria-label="预算决策"
      className="rounded-xl border border-amber-500/40 bg-amber-500/5 p-5"
    >
      <h2 className="text-lg font-semibold text-amber-200">预算不足以继续</h2>
      <dl className="mt-4 grid grid-cols-2 gap-3 text-sm">
        <dt className="text-slate-400">已花费</dt>
        <dd className="font-mono">{yuan(gate.spent, 6)}</dd>
        <dt className="text-slate-400">剩余</dt>
        <dd className="font-mono">{yuan(gate.remaining, 6)}</dd>
        <dt className="text-slate-400">收尾预估</dt>
        <dd className="font-mono">{yuan(gate.finalize_estimate, 6)}</dd>
        <dt className="text-slate-400">截断维度</dt>
        <dd className="font-mono">{gate.truncated_dimensions.join(", ") || "无"}</dd>
      </dl>

      {failure?.kind === "overrun" && (
        <p className="mt-3 text-sm text-amber-300">
          收尾需要 {yuan(failure.finalizeEstimate, 6)}，当前仅剩 {yuan(failure.remaining, 6)}。
          {failure.message && <span className="mt-1 block">{failure.message}</span>}
        </p>
      )}
      {failure?.kind === "field" && (
        <p role="alert" className="mt-3 text-sm text-rose-300">
          {failure.message}
        </p>
      )}
      {failure?.kind === "message" && (
        <p role="alert" className="mt-3 text-sm text-rose-300">{failure.message}</p>
      )}

      <div className="mt-4 flex flex-wrap gap-2">
        {needsConfirmation ? (
          <button
            disabled={busy}
            onClick={() => void submit({ action: "summarize", confirm_overrun: true })}
            className="rounded-lg border border-amber-500/60 px-3 py-2 text-sm text-amber-200"
          >
            确认超支并收尾
          </button>
        ) : (
          <button
            disabled={busy}
            onClick={() => void submit({ action: "summarize" })}
            className="rounded-lg border border-slate-700 px-3 py-2 text-sm text-slate-300"
          >
            用现有结果收尾
          </button>
        )}
      </div>

      <div className="mt-5 border-t border-slate-800 pt-4">
        <p className="text-sm font-medium text-slate-200">追加预算并继续</p>
        <div className="mt-3 grid gap-3 sm:grid-cols-3">
          <label className="text-xs text-slate-400">
            成本上限（¥）
            <input
              type="number"
              min="0"
              step="0.000001"
              value={extension.max_cost_usd}
              onChange={(event) =>
                setExtension({ ...extension, max_cost_usd: Number(event.target.value) })
              }
              className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-2 py-1 font-mono text-sm text-slate-200"
            />
          </label>
          <label className="text-xs text-slate-400">
            时长上限 (秒)
            <input
              type="number"
              min="0"
              step="1"
              value={extension.max_wall_clock_s}
              onChange={(event) =>
                setExtension({ ...extension, max_wall_clock_s: Number(event.target.value) })
              }
              className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-2 py-1 font-mono text-sm text-slate-200"
            />
          </label>
          <label className="text-xs text-slate-400">
            目标请求上限
            <input
              type="number"
              min="0"
              step="1"
              value={extension.max_target_requests}
              onChange={(event) =>
                setExtension({ ...extension, max_target_requests: Number(event.target.value) })
              }
              className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-2 py-1 font-mono text-sm text-slate-200"
            />
          </label>
        </div>
        <button
          disabled={busy}
          onClick={extend}
          className="mt-3 rounded-lg border border-emerald-500/40 px-3 py-2 text-sm text-emerald-300"
        >
          追加并继续
        </button>
      </div>
    </section>
  );
}
