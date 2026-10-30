import { useQuery } from "@tanstack/react-query";
import { useParams } from "react-router-dom";

import { api } from "../api/client";
import { EventTimeline } from "../components/EventTimeline";
import { StatusBadge } from "../components/StatusBadge";
import { useRunEvents } from "../hooks/useRunEvents";
import { formatDate, score } from "../lib/format";

export function PublicRunPage() {
  const { runId = "" } = useParams();
  const run = useQuery({ queryKey: ["public-run", runId], queryFn: () => api.publicRun(runId) });
  const report = useQuery({
    queryKey: ["public-report", runId],
    queryFn: () => api.publicReport(runId),
    retry: false,
  });
  const { events, connected } = useRunEvents(runId, true);

  if (run.isLoading) {
    return <main className="grid min-h-screen place-items-center text-slate-400">加载中…</main>;
  }
  if (run.error || !run.data) {
    return <main className="grid min-h-screen place-items-center text-slate-400">该运行未公开或不存在。</main>;
  }

  return (
    <main className="min-h-screen bg-slate-950">
      <div className="mx-auto max-w-4xl px-6 py-10">
        <p className="text-xs uppercase tracking-[0.3em] text-emerald-400">Eva Agent · 公开报告</p>
        <h1 className="mt-2 flex items-center gap-3 text-3xl font-semibold">
          {run.data.run_id} <StatusBadge status={run.data.status} />
        </h1>
        <p className="mt-2 text-sm text-slate-400">创建于 {formatDate(run.data.created_at)}</p>

        {report.data && (
          <>
            <p className="mt-8 text-sm text-slate-400">
              综合得分 <span className="font-mono text-emerald-300">{score(report.data.report.overall_score)}</span>
              · 报告版本 v{report.data.version}
            </p>
            <div className="mt-4 grid gap-3 sm:grid-cols-2">
              {(report.data.report.dimensions ?? []).map((dimension) => (
                <article key={dimension.dimension} className="rounded-xl border border-slate-800 bg-slate-900/40 p-4">
                  <p className="text-sm font-medium">{dimension.dimension}</p>
                  <p className="mt-1 font-mono text-xs text-slate-400">
                    {dimension.status} · {score(dimension.score)}
                  </p>
                  <p className="mt-2 text-xs text-slate-500">{dimension.summary}</p>
                </article>
              ))}
            </div>
            <pre className="mt-6 max-h-[32rem] overflow-auto whitespace-pre-wrap rounded-xl border border-slate-800 bg-slate-900/40 p-5 text-xs text-slate-400">
              {report.data.markdown}
            </pre>
          </>
        )}

        <div className="mt-8">
          <EventTimeline events={events} connected={connected} />
        </div>
        <p className="mt-8 text-center text-xs text-slate-600">
          评测由 Eva Agent 生成；公开页只展示已发布、已完成运行的脱敏结果。
        </p>
      </div>
    </main>
  );
}
