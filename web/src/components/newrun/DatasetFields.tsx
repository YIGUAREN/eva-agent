import type { NewRunState } from "../../lib/newRun";

interface Props {
  value: NewRunState["dataset"];
  pathError?: string;
  columnMapError?: string;
  onChange: (next: NewRunState["dataset"]) => void;
}

export function DatasetFields({ value, pathError, columnMapError, onChange }: Props) {
  return (
    <>
      <label className="mt-6 block text-sm text-slate-300">
        数据集文件路径
        <input
          type="text"
          required
          value={value.path}
          onChange={(event) => onChange({ ...value, path: event.target.value })}
          placeholder="data/datasets/qa.csv"
          className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm outline-none focus:border-emerald-500"
        />
      </label>
      <p className="mt-1 text-xs text-slate-500">
        路径相对于运行 eva-web 的机器；worker 侧会再次校验。
      </p>
      {pathError && <p className="mt-1 text-xs text-rose-400">{pathError}</p>}
      <label className="mt-4 block text-sm text-slate-300">
        格式
        <select
          value={value.format}
          onChange={(event) =>
            onChange({ ...value, format: event.target.value as NewRunState["dataset"]["format"] })
          }
          className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm outline-none focus:border-emerald-500"
        >
          <option value="csv">CSV</option>
          <option value="jsonl">JSONL</option>
        </select>
      </label>
      <label className="mt-4 block text-sm text-slate-300">
        列映射（可选）
        <input
          type="text"
          value={value.columnMap}
          onChange={(event) => onChange({ ...value, columnMap: event.target.value })}
          placeholder="query=q,response=a"
          className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm outline-none focus:border-emerald-500"
        />
      </label>
      {columnMapError && <p className="mt-1 text-xs text-rose-400">{columnMapError}</p>}
    </>
  );
}
