import { useMutation, useQuery } from "@tanstack/react-query";
import type { FormEvent } from "react";
import { useState } from "react";
import { useNavigate } from "react-router-dom";

import { api } from "../api/client";

export function NewRunPage() {
  const navigate = useNavigate();
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState("");
  const [context, setContext] = useState("");
  const [cost, setCost] = useState("1");
  const [seconds, setSeconds] = useState("300");
  const [requests, setRequests] = useState("5");

  const catalog = useQuery({ queryKey: ["graders"], queryFn: api.graders });
  const targets = useQuery({ queryKey: ["targets"], queryFn: api.demoTargets });

  const create = useMutation({
    mutationFn: () =>
      api.createRun({
        input: {
          kind: "paste",
          paste: {
            query,
            ...(response ? { response } : {}),
            ...(context ? { context } : {}),
          },
        },
        budget: {
          max_cost_usd: Number(cost),
          max_wall_clock_s: Number(seconds),
          max_target_requests: Number(requests),
        },
      }),
    onSuccess: (data) => navigate(`/runs/${data.run_id}`),
  });

  function submit(event: FormEvent) {
    event.preventDefault();
    create.mutate();
  }

  return (
    <section className="grid gap-6 lg:grid-cols-[2fr_1fr]">
      <form onSubmit={submit} className="rounded-xl border border-slate-800 bg-slate-900/40 p-6">
        <h1 className="text-2xl font-semibold">新建评估</h1>
        <p className="mt-1 text-sm text-slate-400">粘贴待评估的问答，预算为硬上限。</p>

        <label className="mt-6 block text-sm text-slate-300">
          问题 / 提示词
          <textarea
            required
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            className="mt-2 h-28 w-full rounded-lg border border-slate-700 bg-slate-950 p-3 text-sm outline-none focus:border-emerald-500"
          />
        </label>
        <label className="mt-4 block text-sm text-slate-300">
          待评估回答
          <textarea
            value={response}
            onChange={(event) => setResponse(event.target.value)}
            className="mt-2 h-40 w-full rounded-lg border border-slate-700 bg-slate-950 p-3 text-sm outline-none focus:border-emerald-500"
          />
        </label>
        <label className="mt-4 block text-sm text-slate-300">
          背景（可选）
          <textarea
            value={context}
            onChange={(event) => setContext(event.target.value)}
            className="mt-2 h-20 w-full rounded-lg border border-slate-700 bg-slate-950 p-3 text-sm outline-none focus:border-emerald-500"
          />
        </label>

        <div className="mt-6 grid gap-4 sm:grid-cols-3">
          <label className="block text-sm text-slate-300">
            成本上限（USD）
            <input type="number" min="0.01" step="0.01" value={cost} onChange={(event) => setCost(event.target.value)} className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2" />
          </label>
          <label className="block text-sm text-slate-300">
            时长上限（秒）
            <input type="number" min="1" step="1" value={seconds} onChange={(event) => setSeconds(event.target.value)} className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2" />
          </label>
          <label className="block text-sm text-slate-300">
            目标请求上限
            <input type="number" min="1" step="1" value={requests} onChange={(event) => setRequests(event.target.value)} className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2" />
          </label>
        </div>

        {create.error && <p className="mt-4 text-sm text-rose-400">{create.error.message}</p>}
        <button
          disabled={create.isPending || !query.trim()}
          className="mt-6 rounded-lg bg-emerald-500 px-5 py-2.5 font-medium text-slate-950 disabled:opacity-50"
        >
          {create.isPending ? "提交中…" : "排队运行"}
        </button>
      </form>

      <aside className="space-y-6">
        <section className="rounded-xl border border-slate-800 bg-slate-900/40 p-5">
          <h2 className="text-sm font-semibold text-slate-300">可用 grader</h2>
          <div className="mt-3 flex flex-wrap gap-2">
            {(catalog.data?.graders ?? []).map((grader) => (
              <span key={grader.id} className="rounded-full bg-slate-800 px-2.5 py-1 text-xs text-slate-300">
                {grader.id}
              </span>
            ))}
          </div>
        </section>
        <section className="rounded-xl border border-slate-800 bg-slate-900/40 p-5">
          <h2 className="text-sm font-semibold text-slate-300">演示目标</h2>
          <ul className="mt-3 space-y-2 text-sm text-slate-400">
            {(targets.data?.targets ?? []).map((target) => (
              <li key={target.name} className="flex justify-between">
                <span className="text-slate-200">{target.name}</span>
                <span className="font-mono text-xs">{target.kind}</span>
              </li>
            ))}
          </ul>
        </section>
      </aside>
    </section>
  );
}
