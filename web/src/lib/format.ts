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
