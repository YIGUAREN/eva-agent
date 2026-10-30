import { useMutation, useQuery } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";

import { api } from "../api/client";
import { EventTimeline } from "../components/EventTimeline";
import { PlanEditor } from "../components/PlanEditor";
import { StatusBadge } from "../components/StatusBadge";
import { useRunEvents } from "../hooks/useRunEvents";
import { formatDate, isTerminal, score } from "../lib/format";

export function RunDetailPage() {
  const { runId = "" } = useParams();
  const detail = useQuery({
    queryKey: ["run", runId],
    queryFn: () => api.getRun(runId),
    refetchInterval: (query) => (query.state.data && isTerminal(query.state.data.status) ? false : 2000),
  });
  const { events, connected } = useRunEvents(runId);
  const run = detail.data;
  const report = useQuery({
    queryKey: ["report", runId],
    queryFn: () => api.report(runId),
    retry: 3,
    retryDelay: 1000,
    enabled: run?.status === "completed",
  });
  const approve = useMutation({
    mutationFn: (plan: Record<string, unknown> | null) => api.approve(runId, true, plan ?? undefined),
    onSuccess: () => detail.refetch(),
  });
  const reject = useMutation({
    mutationFn: () => api.approve(runId, false),
    onSuccess: () => detail.refetch(),
  });
  const cancel = useMutation({ mutationFn: () => api.cancel(runId), onSuccess: () => detail.refetch() });
  const publish = useMutation({
    mutationFn: (published: boolean) => api.publish(runId, published),
    onSuccess: () => detail.refetch(),
  });

  const actionError = approve.error ?? reject.error ?? cancel.error ?? publish.error;

  function runAction(action: () => void) {
    approve.reset();
    reject.reset();
    cancel.reset();
    publish.reset();
    action();
  }

  if (detail.isLoading) {
    return <p className="text-slate-400">加载中…</p>;
  }
  if (detail.error || !run) {
    return <p className="text-rose-400">{detail.error?.message ?? "运行不存在"}</p>;
  }

  function downloadMarkdown() {
    const markdown = report.data?.markdown ?? "";
    const blob = new Blob([markdown], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${runId}-report.md`;
    link.click();
    URL.revokeObjectURL(url);
  }

  return (
    <section className="space-y-6">
      <header className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="font-mono text-sm text-slate-500">{run.run_id}</p>
          <h1 className="mt-1 flex items-center gap-3 text-2xl font-semibold">
            运行详情 <StatusBadge status={run.status} />
          </h1>
          <p className="mt-2 text-sm text-slate-400">创建于 {formatDate(run.created_at)}</p>
        </div>
        <div className="flex flex-wrap gap-2">
          {["queued", "waiting_approval"].includes(run.status) && (
            <button onClick={() => runAction(() => cancel.mutate())} className="rounded-lg border border-slate-700 px-3 py-2 text-sm text-slate-300">
              取消
            </button>
          )}
          {run.status === "completed" && (
            <>
              <button onClick={() => runAction(() => publish.mutate(!run.is_public))} className="rounded-lg border border-emerald-500/40 px-3 py-2 text-sm text-emerald-300">
                {run.is_public ? "取消发布" : "发布"}
              </button>
              {run.is_public && (
                <Link to={`/share/${run.run_id}`} className="rounded-lg border border-slate-700 px-3 py-2 text-sm text-slate-300">
                  公开页
                </Link>
              )}
            </>
          )}
        </div>
      </header>

      {actionError && (
        <p role="alert" className="rounded-lg border border-rose-500/30 bg-rose-500/5 p-4 text-sm text-rose-300">
          {actionError.message}
        </p>
      )}

      {run.error && <p className="rounded-lg border border-rose-500/30 bg-rose-500/5 p-4 text-sm text-rose-300">{run.error}</p>}

      {run.status === "waiting_approval" && (
        <PlanEditor
          plan={run.plan}
          busy={approve.isPending || reject.isPending}
          onApprove={(plan) => runAction(() => approve.mutate(plan))}
          onReject={() => runAction(() => reject.mutate())}
        />
      )}

      <div className="grid gap-6 lg:grid-cols-[3fr_2fr]">
        <div className="space-y-6">
          <section className="rounded-xl border border-slate-800 bg-slate-900/40 p-5">
            <h2 className="text-lg font-semibold">输入与预算</h2>
            <dl className="mt-4 grid grid-cols-2 gap-3 text-sm">
              <dt className="text-slate-400">输入类型</dt>
              <dd className="font-mono">{String(run.input.kind ?? "—")}</dd>
              <dt className="text-slate-400">成本上限</dt>
              <dd className="font-mono">{run.budget.max_cost_usd} USD</dd>
              <dt className="text-slate-400">时长上限</dt>
              <dd className="font-mono">{run.budget.max_wall_clock_s} s</dd>
              <dt className="text-slate-400">目标请求</dt>
              <dd className="font-mono">{run.budget.max_target_requests}</dd>
            </dl>
          </section>

          {report.data && (
            <section className="rounded-xl border border-slate-800 bg-slate-900/40 p-5">
              <div className="flex items-center justify-between">
                <h2 className="text-lg font-semibold">评估报告</h2>
                <button onClick={downloadMarkdown} className="text-sm text-emerald-400">下载 Markdown</button>
              </div>
              <p className="mt-3 text-sm text-slate-400">
                综合得分 <span className="font-mono text-emerald-300">{score(report.data.report.overall_score)}</span>
                · 报告版本 v{report.data.version}
              </p>
              <div className="mt-4 grid gap-3 sm:grid-cols-2">
                {(report.data.report.dimensions ?? []).map((dimension) => (
                  <article key={dimension.dimension} className="rounded-lg border border-slate-800 bg-slate-950 p-3">
                    <p className="text-sm font-medium">{dimension.dimension}</p>
                    <p className="mt-1 font-mono text-xs text-slate-400">
                      {dimension.status} · {score(dimension.score)}
                    </p>
                    <p className="mt-2 text-xs text-slate-500">{dimension.summary}</p>
                  </article>
                ))}
              </div>
              <pre className="mt-5 max-h-72 overflow-auto whitespace-pre-wrap rounded-lg bg-slate-950 p-4 text-xs text-slate-400">
                {report.data.markdown}
              </pre>
            </section>
          )}
        </div>
        <EventTimeline events={events} connected={connected} />
      </div>
    </section>
  );
}
