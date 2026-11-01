import { useQuery } from "@tanstack/react-query";
import type { FormEvent } from "react";
import { useState } from "react";
import { useNavigate } from "react-router-dom";

import { ApiError, api } from "../api/client";
import type { GraderInfo } from "../api/types";
import { DatasetFields } from "../components/newrun/DatasetFields";
import { PasteFields } from "../components/newrun/PasteFields";
import { TargetFields } from "../components/newrun/TargetFields";
import { UploadField } from "../components/newrun/UploadField";
import {
  INITIAL_NEW_RUN,
  MAX_AUTHORIZATION_REF,
  hasErrors,
  submitNewRun,
  validateNewRun,
} from "../lib/newRun";
import type { InputKind, NewRunState } from "../lib/newRun";
import { allGraderIds, partitionGraders, toGraderSelection, toggleGrader } from "../lib/graders";

const KINDS: { value: InputKind; label: string }[] = [
  { value: "paste", label: "粘贴" },
  { value: "dataset", label: "数据集" },
  { value: "upload", label: "上传证据包" },
  { value: "target", label: "在线目标" },
];

export function NewRunPage() {
  const navigate = useNavigate();
  const [state, setState] = useState<NewRunState>(INITIAL_NEW_RUN);
  const [submitting, setSubmitting] = useState<"uploading" | "creating" | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const catalog = useQuery({ queryKey: ["graders"], queryFn: api.graders });
  const targets = useQuery({ queryKey: ["targets"], queryFn: api.demoTargets });

  const graders = catalog.data?.graders ?? [];
  const groups = partitionGraders(graders);
  // ``null`` means the user has not touched the list yet: everything is checked.
  const [selected, setSelected] = useState<Set<string> | null>(null);
  const checked = selected ?? allGraderIds(graders);
  const noneChecked = catalog.isSuccess && checked.size === 0;
  const errors = validateNewRun(state);
  const invalid = hasErrors(errors) || noneChecked;

  function update(patch: Partial<NewRunState>) {
    setState((current) => ({ ...current, ...patch }));
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (invalid || submitting !== null) {
      return;
    }
    setSubmitError(null);
    const selection = catalog.isSuccess ? toGraderSelection(graders, checked) : undefined;
    try {
      setSubmitting(state.kind === "upload" ? "uploading" : "creating");
      const result = await submitNewRun(state, selection, {
        upload: async (file) => {
          const uploaded = await api.upload(file);
          setSubmitting("creating");
          return uploaded;
        },
        createRun: api.createRun,
      });
      navigate(`/runs/${result.run_id}`);
    } catch (error) {
      setSubmitError(
        error instanceof ApiError && error.status === 503
          ? "解压服务不可用，请确认 sandbox-runner 已启动"
          : error instanceof Error
            ? error.message
            : "请求失败，请重试",
      );
      setSubmitting(null);
    }
  }

  return (
    <section className="grid gap-6 lg:grid-cols-[2fr_1fr]">
      <form onSubmit={submit} className="rounded-xl border border-slate-800 bg-slate-900/40 p-6">
        <h1 className="text-2xl font-semibold">新建评估</h1>
        <p className="mt-1 text-sm text-slate-400">选择输入类型，预算为硬上限。</p>

        <div className="mt-5 flex flex-wrap gap-2" role="group" aria-label="输入类型">
          {KINDS.map((kind) => (
            <button
              key={kind.value}
              type="button"
              aria-pressed={state.kind === kind.value}
              onClick={() => update({ kind: kind.value })}
              className={
                state.kind === kind.value
                  ? "rounded-lg bg-emerald-500 px-3 py-1.5 text-sm font-medium text-slate-950"
                  : "rounded-lg border border-slate-700 px-3 py-1.5 text-sm text-slate-300 hover:border-emerald-500"
              }
            >
              {kind.label}
            </button>
          ))}
        </div>

        {state.kind === "paste" && (
          <PasteFields
            value={state.paste}
            queryError={errors.pasteQuery}
            onChange={(paste) => update({ paste })}
          />
        )}
        {state.kind === "dataset" && (
          <DatasetFields
            value={state.dataset}
            pathError={errors.datasetPath}
            columnMapError={errors.datasetColumnMap}
            onChange={(dataset) => update({ dataset })}
          />
        )}
        {state.kind === "upload" && (
          <UploadField
            file={state.upload.file}
            error={errors.uploadFile}
            onChange={(file) => update({ upload: { file } })}
          />
        )}
        {state.kind === "target" && (
          <TargetFields
            targets={targets.data?.targets ?? []}
            loading={targets.isLoading}
            loadError={targets.error?.message}
            targetRef={state.target.targetRef}
            prompts={state.target.prompts}
            targetRefError={errors.targetRef}
            promptsError={errors.targetPrompts}
            onTargetChange={(targetRef) => update({ target: { ...state.target, targetRef } })}
            onPromptsChange={(prompts) => update({ target: { ...state.target, prompts } })}
          />
        )}

        <div className="mt-6 grid gap-4 sm:grid-cols-3">
          <label className="block text-sm text-slate-300">
            成本上限（¥）
            <input
              type="number"
              required
              min="0.01"
              step="0.01"
              value={state.budget.cost}
              onChange={(event) => update({ budget: { ...state.budget, cost: event.target.value } })}
              className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2"
            />
          </label>
          <label className="block text-sm text-slate-300">
            时长上限（秒）
            <input
              type="number"
              required
              min="1"
              step="1"
              value={state.budget.seconds}
              onChange={(event) =>
                update({ budget: { ...state.budget, seconds: event.target.value } })
              }
              className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2"
            />
          </label>
          <label className="block text-sm text-slate-300">
            目标请求上限
            <input
              type="number"
              required
              min="1"
              step="1"
              value={state.budget.requests}
              onChange={(event) =>
                update({ budget: { ...state.budget, requests: event.target.value } })
              }
              className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2"
            />
          </label>
        </div>
        {errors.budget && <p className="mt-1 text-xs text-rose-400">{errors.budget}</p>}

        <details className="mt-5 rounded-lg border border-slate-800 bg-slate-950/40 p-3">
          <summary className="cursor-pointer text-sm text-slate-300">高级选项</summary>
          <label className="mt-3 block text-sm text-slate-300">
            授权引用（可选）
            <input
              type="text"
              value={state.authorizationRef}
              maxLength={MAX_AUTHORIZATION_REF}
              onChange={(event) => update({ authorizationRef: event.target.value })}
              placeholder="例如 incident-2026-001"
              className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm outline-none focus:border-emerald-500"
            />
          </label>
          <p className="mt-1 text-xs text-slate-500">
            留空时后端使用安全策略中的默认值；该值会写入审计记录。
          </p>
          {errors.authorizationRef && (
            <p className="mt-1 text-xs text-rose-400">{errors.authorizationRef}</p>
          )}
        </details>

        {submitError && <p className="mt-4 text-sm text-rose-400">{submitError}</p>}
        <button
          disabled={submitting !== null || invalid}
          className="mt-6 rounded-lg bg-emerald-500 px-5 py-2.5 font-medium text-slate-950 disabled:opacity-50"
        >
          {submitting === "uploading"
            ? "上传解压中…"
            : submitting === "creating"
              ? "创建中…"
              : "排队运行"}
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
