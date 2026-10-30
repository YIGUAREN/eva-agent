import { useState } from "react";

export function PlanEditor({
  plan,
  busy,
  onApprove,
  onReject,
}: {
  plan: Record<string, unknown> | null;
  busy: boolean;
  onApprove: (plan: Record<string, unknown> | null) => void;
  onReject: () => void;
}) {
  const [draft, setDraft] = useState(plan ? JSON.stringify(plan, null, 2) : "");
  const [error, setError] = useState<string | null>(null);

  function approve() {
    if (!draft.trim()) {
      onApprove(null);
      return;
    }
    try {
      const parsed = JSON.parse(draft) as unknown;
      if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
        setError("计划必须是一个 JSON 对象");
        return;
      }
      setError(null);
      onApprove(parsed as Record<string, unknown>);
    } catch {
      setError("计划不是合法 JSON");
    }
  }

  return (
    <section className="rounded-xl border border-amber-500/30 bg-amber-500/5 p-5">
      <h2 className="text-lg font-semibold text-amber-200">待审批计划</h2>
      <p className="mt-1 text-sm text-slate-400">
        可以直接批准，也可以编辑 JSON 后批准；编辑会重新校验 grader 与维度。
      </p>
      <textarea
        value={draft}
        onChange={(event) => setDraft(event.target.value)}
        spellCheck={false}
        className="mt-4 h-72 w-full rounded-lg border border-slate-700 bg-slate-950 p-3 font-mono text-xs text-slate-200 outline-none focus:border-amber-400"
      />
      {error && <p className="mt-2 text-sm text-rose-400">{error}</p>}
      <div className="mt-4 flex gap-3">
        <button
          onClick={approve}
          disabled={busy}
          className="rounded-lg bg-emerald-500 px-4 py-2 text-sm font-medium text-slate-950 disabled:opacity-50"
        >
          批准并执行
        </button>
        <button
          onClick={onReject}
          disabled={busy}
          className="rounded-lg border border-rose-500/40 px-4 py-2 text-sm text-rose-300 disabled:opacity-50"
        >
          拒绝
        </button>
      </div>
    </section>
  );
}
