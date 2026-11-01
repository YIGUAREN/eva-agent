import type { DemoTarget } from "../../api/types";

interface Props {
  targets: DemoTarget[];
  loading: boolean;
  loadError?: string;
  targetRef: string;
  prompts: string;
  targetRefError?: string;
  promptsError?: string;
  onTargetChange: (value: string) => void;
  onPromptsChange: (value: string) => void;
}

export function TargetFields({
  targets,
  loading,
  loadError,
  targetRef,
  prompts,
  targetRefError,
  promptsError,
  onTargetChange,
  onPromptsChange,
}: Props) {
  const promptCount = prompts.split(/\r?\n/).filter((line) => line.trim().length > 0).length;
  return (
    <>
      <label className="mt-6 block text-sm text-slate-300">
        目标
        <select
          value={targetRef}
          onChange={(event) => onTargetChange(event.target.value)}
          className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm outline-none focus:border-emerald-500"
        >
          <option value="">请选择目标</option>
          {targets.map((target) => (
            <option key={target.name} value={target.name}>
              {target.name} · {target.kind} · {target.model}
            </option>
          ))}
        </select>
      </label>
      {loading && <p className="mt-1 text-xs text-slate-500">加载目标中…</p>}
      {loadError && <p className="mt-1 text-xs text-rose-400">{loadError}</p>}
      {!loading && !loadError && targets.length === 0 && (
        <p className="mt-1 text-xs text-amber-400">暂无可用目标，请检查 config/targets.yaml</p>
      )}
      <p className="mt-1 text-xs text-slate-500">
        目标来自 config/targets.yaml，请确认目标服务可达。
      </p>
      {targetRefError && <p className="mt-1 text-xs text-rose-400">{targetRefError}</p>}
      <label className="mt-4 block text-sm text-slate-300">
        提示词（每行一个）
        <textarea
          value={prompts}
          onChange={(event) => onPromptsChange(event.target.value)}
          className="mt-2 h-40 w-full rounded-lg border border-slate-700 bg-slate-950 p-3 text-sm outline-none focus:border-emerald-500"
        />
      </label>
      <p className="mt-1 text-xs text-slate-500">
        已填写 {promptCount} 条；最多 100 条，单条不超过 16384 字节。
      </p>
      {promptsError && <p className="mt-1 text-xs text-rose-400">{promptsError}</p>}
    </>
  );
}
