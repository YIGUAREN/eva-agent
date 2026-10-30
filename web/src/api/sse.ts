import type { RunEvent } from "./types";

export function parseEventData(raw: string): RunEvent | null {
  try {
    const value = JSON.parse(raw) as { seq?: unknown; type?: unknown; payload?: unknown };
    if (typeof value.seq !== "number" || typeof value.type !== "string") {
      return null;
    }
    const payload =
      value.payload !== null && typeof value.payload === "object"
        ? (value.payload as Record<string, unknown>)
        : {};
    return { seq: value.seq, type: value.type, payload };
  } catch {
    return null;
  }
}

export function mergeEvent(existing: RunEvent[], event: RunEvent): RunEvent[] {
  if (existing.some((item) => item.seq === event.seq)) {
    return existing;
  }
  return [...existing, event].sort((left, right) => left.seq - right.seq);
}
