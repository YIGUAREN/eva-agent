import type { NewRunState } from "../../lib/newRun";

interface Props {
  value: NewRunState["paste"];
  queryError?: string;
  onChange: (next: NewRunState["paste"]) => void;
}

export function PasteFields({ value, queryError, onChange }: Props) {
  return (
    <>
      <label className="mt-6 block text-sm text-slate-300">
        问题 / 提示词
        <textarea
          required
          value={value.query}
          onChange={(event) => onChange({ ...value, query: event.target.value })}
          className="mt-2 h-28 w-full rounded-lg border border-slate-700 bg-slate-950 p-3 text-sm outline-none focus:border-emerald-500"
        />
      </label>
      {queryError && <p className="mt-1 text-xs text-rose-400">{queryError}</p>}
      <label className="mt-4 block text-sm text-slate-300">
        待评估回答
        <textarea
          value={value.response}
          onChange={(event) => onChange({ ...value, response: event.target.value })}
          className="mt-2 h-40 w-full rounded-lg border border-slate-700 bg-slate-950 p-3 text-sm outline-none focus:border-emerald-500"
        />
      </label>
      <label className="mt-4 block text-sm text-slate-300">
        背景（可选）
        <textarea
          value={value.context}
          onChange={(event) => onChange({ ...value, context: event.target.value })}
          className="mt-2 h-20 w-full rounded-lg border border-slate-700 bg-slate-950 p-3 text-sm outline-none focus:border-emerald-500"
        />
      </label>
    </>
  );
}
