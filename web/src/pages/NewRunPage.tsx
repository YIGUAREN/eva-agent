import { useMutation, useQuery } from "@tanstack/react-query";
import type { FormEvent } from "react";
import { useState } from "react";
import { useNavigate } from "react-router-dom";

import { api } from "../api/client";
import type { GraderInfo } from "../api/types";
import { allGraderIds, partitionGraders, toGraderSelection, toggleGrader } from "../lib/graders";

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

  const graders = catalog.data?.graders ?? [];
  const groups = partitionGraders(graders);
  // ``null`` means the user has not touched the list yet: everything is checked.
  const [selected, setSelected] = useState<Set<string> | null>(null);
  const checked = selected ?? allGraderIds(graders);
  const noneChecked = catalog.isSuccess && checked.size === 0;

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
          // An explicit selection only goes out once the catalog is known;
          // otherwise the planner keeps its default choice.
          ...(catalog.isSuccess ? { graders: toGraderSelection(graders, checked) } : {}),
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
            成本上限（¥）
            <input type="number" required min="0.01" step="0.01" value={cost} onChange={(event) => setCost(event.target.value)} className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2" />
          </label>
          <label className="block text-sm text-slate-300">
            时长上限（秒）
            <input type="number" required min="1" step="1" value={seconds} onChange={(event) => setSeconds(event.target.value)} className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2" />
          </label>
          <label className="block text-sm text-slate-300">
            目标请求上限
            <input type="number" required min="1" step="1" value={requests} onChange={(event) => setRequests(event.target.value)} className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2" />
          </label>
        </div>

        {create.error && <p className="mt-4 text-sm text-rose-400">{create.error.message}</p>}
        <button
          disabled={create.isPending || !query.trim() || noneChecked}
          className="mt-6 rounded-lg bg-emerald-500 px-5 py-2.5 font-medium text-slate-950 disabled:opacity-50"
        >
          {create.isPending ? "提交中…" : "排队运行"}
        </button>
      </form>

      <aside className="space-y-6">
        <section className="rounded-xl border border-slate-800 bg-slate-900/40 p-5">
          <h2 className="text-sm font-semibold text-slate-300">评分器</h2>
          <p className="mt-1 text-xs text-slate-500">只运行勾选的评分器，未勾选的不参与评分。</p>
          {catalog.isLoading && <p className="mt-3 text-xs text-slate-500">加载中…</p>}
          {catalog.error && <p className="mt-3 text-xs text-rose-400">{catalog.error.message}</p>}
          <GraderGroup
            title="质量"
            graders={groups.quality}
            checked={checked}
            onToggle={(id) => setSelected(toggleGrader(checked, id))}
          />
          <GraderGroup
            title="安全"
            graders={groups.security}
            checked={checked}
            onToggle={(id) => setSelected(toggleGrader(checked, id))}
          />
          {noneChecked && <p className="mt-3 text-xs text-amber-400">至少勾选一个评分器。</p>}
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

function GraderGroup({
  title,
  graders,
  checked,
  onToggle,
}: {
  title: string;
  graders: GraderInfo[];
  checked: ReadonlySet<string>;
  onToggle: (id: string) => void;
}) {
  if (graders.length === 0) {
    return null;
  }
  return (
    <div className="mt-4">
      <h3 className="text-xs font-medium text-slate-400">{title}</h3>
      <ul className="mt-2 space-y-1.5">
        {graders.map((grader) => (
          <li key={grader.id}>
            <label className="flex cursor-pointer items-center gap-2 text-sm text-slate-200">
              <input
                type="checkbox"
                checked={checked.has(grader.id)}
                onChange={() => onToggle(grader.id)}
                className="size-4 accent-emerald-500"
              />
              <span className="font-mono text-xs">{grader.id}</span>
              <span className="text-xs text-slate-500">{grader.cost}</span>
            </label>
          </li>
        ))}
      </ul>
    </div>
  );
}
