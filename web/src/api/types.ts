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

export interface Budget {
  max_cost_usd: number;
  max_wall_clock_s: number;
  max_target_requests: number;
}

export interface RunUsageView {
  spent_usd: number;
  reserved_usd: number;
  spent_wall_clock_s: number;
  target_requests: number;
  aig_requests: number;
}

export interface BudgetGate {
  finalize_estimate: number;
  remaining: number;
  spent: number;
  truncated_dimensions: string[];
  seq?: number;
}

export type BudgetDecisionBody =
  | { action: "summarize"; confirm_overrun?: boolean }
  | { action: "extend"; budget: Budget };

export interface RunDetail extends RunSummary {
  input: Record<string, unknown>;
  budget: Budget;
  plan: Record<string, unknown> | null;
  error: string | null;
  is_public: boolean;
  created_by: string | null;
  usage: RunUsageView | null;
  budget_gate: BudgetGate | null;
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
    usage?: {
      spent_usd?: number;
      estimated?: boolean;
      cached?: number;
      fresh?: number;
      budget_enforced?: boolean;
    };
    [key: string]: unknown;
  };
}

export interface GraderPackageInfo {
  package_id: string;
  run_id: string;
  sha256: string;
  size_bytes: number;
  created_at?: string;
  replayable?: boolean;
  replay_blockers?: string[];
  members?: string[];
  download_url?: string;
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

export interface GraderSelection {
  quality: string[];
  security: string[];
}

export interface CreateRunBody {
  input: {
    kind: "paste";
    paste: { query: string; response?: string; context?: string };
    graders?: GraderSelection;
  };
  budget: Budget;
  idempotency_key?: string;
}
