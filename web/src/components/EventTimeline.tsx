import type { RunEvent } from "../api/types";

export function EventTimeline({ events, connected }: { events: RunEvent[]; connected: boolean }) {
  return (
    <section className="rounded-xl border border-slate-800 bg-slate-900/40 p-5">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">运行事件</h2>
        <span className={`text-xs ${connected ? "text-emerald-400" : "text-slate-500"}`}>
          {connected ? "实时连接" : "等待连接"}
        </span>
      </div>
      <ol className="mt-4 space-y-2">
        {events.length === 0 && <li className="text-sm text-slate-500">还没有事件。</li>}
        {events
          .slice(-40)
          .reverse()
          .map((event) => (
            <li key={event.seq} className="flex gap-3 border-b border-slate-800/60 pb-2 text-sm last:border-none">
              <span className="w-10 shrink-0 font-mono text-xs text-slate-500">#{event.seq}</span>
              <div>
                <p className="font-medium text-slate-200">{event.type}</p>
                {Object.keys(event.payload).length > 0 && (
                  <p className="mt-0.5 break-all font-mono text-xs text-slate-500">
                    {JSON.stringify(event.payload)}
                  </p>
                )}
              </div>
            </li>
          ))}
      </ol>
    </section>
  );
}
