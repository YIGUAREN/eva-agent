const TONE: Record<string, string> = {
  queued: "bg-slate-700/60 text-slate-200",
  running: "bg-sky-500/15 text-sky-300",
  waiting_approval: "bg-amber-500/15 text-amber-300",
  waiting_budget: "bg-amber-500/15 text-amber-300",
  completed: "bg-emerald-500/15 text-emerald-300",
  failed: "bg-rose-500/15 text-rose-300",
  cancelled: "bg-slate-600/40 text-slate-300",
};

const LABEL: Record<string, string> = {
  queued: "排队中",
  running: "运行中",
  waiting_approval: "待审批",
  waiting_budget: "待预算确认",
  completed: "已完成",
  failed: "失败",
  cancelled: "已取消",
};

export function StatusBadge({ status }: { status: string }) {
  return (
    <span className={`inline-flex rounded-full px-2.5 py-1 text-xs font-medium ${TONE[status] ?? TONE.queued}`}>
      {LABEL[status] ?? status}
    </span>
  );
}
