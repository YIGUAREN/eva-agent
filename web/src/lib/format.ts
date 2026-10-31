export function formatDate(value: string | null): string {
  if (!value) {
    return "—";
  }
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

export function isTerminal(status: string): boolean {
  return status === "completed" || status === "failed" || status === "cancelled";
}

export function score(value: number | null | undefined): string {
  return value === null || value === undefined ? "n/a" : value.toFixed(3);
}

/** Format an amount with the RMB symbol; the number itself is not converted. */
export function yuan(value: number, digits = 2): string {
  if (!Number.isFinite(value)) return "—";
  return `¥${value.toFixed(digits)}`;
}

export function spend(value: number | null | undefined): string | null {
  if (value === null || value === undefined || !Number.isFinite(value)) return null;
  return yuan(value);
}
