export type RunStatus =
  | "queued"
  | "running"
  | "waiting_approval"
  | "waiting_budget"
  | "completed"
  | "failed"
  | "cancelled";

export interface RunSummary {
  run_id: string;
  status: RunStatus;
  objective: string | null;
  created_at: string;
  published_at: string | null;
}

export interface RunDetail extends RunSummary {
  input: Record<string, unknown>;
  budget: { max_cost_usd: number; max_wall_clock_s: number; max_target_requests: number };
  plan: Record<string, unknown> | null;
  error: string | null;
  is_public: boolean;
  created_by: string | null;
}

export interface RunEvent {
  seq: number;
  type: string;
  payload: Record<string, unknown>;
}

export interface ReportResponse {
  run_id: string;
  version: number;
  sha256: string;
  markdown: string;
  report: {
    status?: string;
    overall_score?: number | null;
    dimensions?: Array<{ dimension: string; status: string; score?: number | null; summary?: string }>;
    findings?: Array<{ id: string; severity: string; title: string }>;
    [key: string]: unknown;
  };
}

export interface GraderInfo {
  id: string;
  mode: string;
  tags: string[];
  cost: string;
}

export interface DemoTarget {
  name: string;
  kind: string;
  provider: string;
  model: string;
}

export interface CreateRunBody {
  input: { kind: "paste"; paste: { query: string; response?: string; context?: string } };
  budget: { max_cost_usd: number; max_wall_clock_s: number; max_target_requests: number };
  idempotency_key?: string;
}
