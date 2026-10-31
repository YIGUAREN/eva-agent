import type {
  BudgetDecisionBody,
  CreateRunBody,
  DemoTarget,
  GraderInfo,
  ReportResponse,
  RunDetail,
  RunSummary,
} from "./types";

let csrfToken: string | null = null;

export function setCsrfToken(token: string | null): void {
  csrfToken = token;
}

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
    message: string,
    readonly details?: Record<string, unknown>,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  const method = (init.method ?? "GET").toUpperCase();
  if (init.body !== undefined && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  if (csrfToken && method !== "GET" && method !== "HEAD") {
    headers.set("X-CSRF-Token", csrfToken);
  }
  const response = await fetch(path, {
    ...init,
    method,
    headers,
    credentials: "same-origin",
  });
  if (response.status === 204) {
    return undefined as T;
  }
  const body = (await response.json().catch(() => null)) as
    | { error?: { code?: string; message?: string; details?: Record<string, unknown> } }
    | null;
  if (!response.ok) {
    throw new ApiError(
      response.status,
      body?.error?.code ?? "http_error",
      body?.error?.message ?? "request failed",
      body?.error?.details,
    );
  }
  return body as T;
}

export const api = {
  me: () => request<{ csrf_token: string; expires_at: string }>("/api/auth/me"),
  login: (password: string) =>
    request<{ csrf_token: string; expires_at: string }>("/api/auth/login", {
      method: "POST",
      body: JSON.stringify({ password }),
    }),
  logout: () => request<void>("/api/auth/logout", { method: "POST" }),
  listRuns: () => request<{ items: RunSummary[] }>("/api/runs"),
  getRun: (runId: string) => request<RunDetail>(`/api/runs/${runId}`),
  createRun: (body: CreateRunBody) =>
    request<{ run_id: string; status: string }>("/api/runs", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  approve: (runId: string, approved: boolean, plan?: unknown) =>
    request<{ run_id: string }>(`/api/runs/${runId}/approve`, {
      method: "POST",
      body: JSON.stringify(plan === undefined ? { approved } : { approved, plan }),
    }),
  budgetDecision: (runId: string, body: BudgetDecisionBody) =>
    request<{ run_id: string; status: string }>(`/api/runs/${runId}/budget/decision`, {
      method: "POST",
      body: JSON.stringify(body),
    }),
  cancel: (runId: string) => request<{ run_id: string }>(`/api/runs/${runId}/cancel`, { method: "POST" }),
  publish: (runId: string, published: boolean) =>
    request<RunDetail>(`/api/runs/${runId}/publish`, {
      method: "POST",
      body: JSON.stringify({ published }),
    }),
  report: (runId: string) => request<ReportResponse>(`/api/runs/${runId}/report`),
  publicRuns: (cursor?: string) =>
    request<{ items: RunSummary[]; next_cursor: string | null }>(
      cursor ? `/api/public/runs?cursor=${encodeURIComponent(cursor)}` : "/api/public/runs",
    ),
  publicRun: (runId: string) => request<RunSummary>(`/api/public/runs/${runId}`),
  publicReport: (runId: string) => request<ReportResponse>(`/api/public/runs/${runId}/report`),
  graders: () => request<{ graders: GraderInfo[] }>("/api/graders"),
  demoTargets: () => request<{ targets: DemoTarget[] }>("/api/demo-targets"),
};
