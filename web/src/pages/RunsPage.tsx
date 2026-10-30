import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";

import { api } from "../api/client";
import { StatusBadge } from "../components/StatusBadge";
import { formatDate } from "../lib/format";

export function RunsPage() {
  const { data, error, isLoading } = useQuery({
    queryKey: ["runs"],
    queryFn: api.listRuns,
    refetchInterval: 3000,
  });

  return (
    <section>
      <div className="flex items-end justify-between">
        <div>
          <h1 className="text-2xl font-semibold">评估运行</h1>
          <p className="mt-1 text-sm text-slate-400">按创建时间倒序，每 3 秒自动刷新。</p>
        </div>
        <Link to="/new" className="rounded-lg bg-emerald-500 px-4 py-2 text-sm font-medium text-slate-950">
          新建评估
        </Link>
      </div>
      {isLoading && <p className="mt-8 text-slate-400">加载中…</p>}
      {error && <p className="mt-8 text-rose-400">{error.message}</p>}
      <div className="mt-6 overflow-hidden rounded-xl border border-slate-800">
        <table className="w-full text-left text-sm">
          <thead className="bg-slate-900 text-slate-400">
            <tr>
              <th className="px-4 py-3">运行</th>
              <th className="px-4 py-3">状态</th>
              <th className="px-4 py-3">目标</th>
              <th className="px-4 py-3">创建时间</th>
            </tr>
          </thead>
          <tbody>
            {(data?.items ?? []).map((run) => (
              <tr key={run.run_id} className="border-t border-slate-800 hover:bg-slate-900/60">
                <td className="px-4 py-3">
                  <Link className="font-mono text-emerald-400" to={`/runs/${run.run_id}`}>
                    {run.run_id}
                  </Link>
                </td>
                <td className="px-4 py-3">
                  <StatusBadge status={run.status} />
                </td>
                <td className="px-4 py-3 text-slate-300">{run.objective ?? "—"}</td>
                <td className="px-4 py-3 text-slate-400">{formatDate(run.created_at)}</td>
              </tr>
            ))}
            {data && data.items.length === 0 && (
              <tr>
                <td colSpan={4} className="px-4 py-10 text-center text-slate-500">
                  还没有运行
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}
