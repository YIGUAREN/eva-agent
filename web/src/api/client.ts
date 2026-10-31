import type {
  BudgetDecisionBody,
  CreateRunBody,
  DemoTarget,
  GraderInfo,
  GraderPackageInfo,
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

type ErrorEnvelope = {
  error?: { code?: string; message?: string; details?: Record<string, unknown> };
};

function apiError(status: number, body: ErrorEnvelope | null): ApiError {
  return new ApiError(
    status,
    body?.error?.code ?? "http_error",
    body?.error?.message ?? "请求失败，请重试",
    body?.error?.details,
  );
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
  const body = (await response.json().catch(() => null)) as ErrorEnvelope | null;
  if (!response.ok) {
    throw apiError(response.status, body);
  }
  return body as T;
}

async function requestBlob(path: string): Promise<Blob> {
  const response = await fetch(path, { method: "GET", credentials: "same-origin" });
  if (!response.ok) {
    const body = (await response.json().catch(() => null)) as ErrorEnvelope | null;
    throw apiError(response.status, body);
  }
  return response.blob();
}

function graderPackagePath(runId: string, suffix = ""): string {
  return `/api/runs/${encodeURIComponent(runId)}/grader-package${suffix}`;
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
  exportGraderPackage: (runId: string) =>
    request<GraderPackageInfo>(graderPackagePath(runId), { method: "POST" }),
  graderPackage: (runId: string) => request<GraderPackageInfo>(graderPackagePath(runId)),
  graderPackageUrl: (runId: string) => graderPackagePath(runId, "/download"),
  downloadGraderPackage: (runId: string) => requestBlob(graderPackagePath(runId, "/download")),
  publicRuns: (cursor?: string) =>
    request<{ items: RunSummary[]; next_cursor: string | null }>(
      cursor ? `/api/public/runs?cursor=${encodeURIComponent(cursor)}` : "/api/public/runs",
    ),
  publicRun: (runId: string) => request<RunSummary>(`/api/public/runs/${runId}`),
  publicReport: (runId: string) => request<ReportResponse>(`/api/public/runs/${runId}/report`),
  graders: () => request<{ graders: GraderInfo[] }>("/api/graders"),
  demoTargets: () => request<{ targets: DemoTarget[] }>("/api/demo-targets"),
};
