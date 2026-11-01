import type { Budget, CreateRunBody, GraderSelection, RunInputBody } from "../api/types";

export type InputKind = "paste" | "dataset" | "upload" | "target";

export interface NewRunState {
  kind: InputKind;
  paste: { query: string; response: string; context: string };
  dataset: { path: string; format: "csv" | "jsonl"; columnMap: string };
  upload: { file: File | null };
  target: { targetRef: string; prompts: string };
  budget: { cost: string; seconds: string; requests: string };
  authorizationRef: string;
}

export const INITIAL_NEW_RUN: NewRunState = {
  kind: "paste",
  paste: { query: "", response: "", context: "" },
  dataset: { path: "", format: "csv", columnMap: "" },
  upload: { file: null },
  target: { targetRef: "", prompts: "" },
  budget: { cost: "1", seconds: "300", requests: "5" },
  authorizationRef: "",
};

export interface FieldErrors {
  pasteQuery?: string;
  datasetPath?: string;
  datasetColumnMap?: string;
  uploadFile?: string;
  targetRef?: string;
  targetPrompts?: string;
  budget?: string;
  authorizationRef?: string;
}

export const MAX_PROMPTS = 100;
export const MAX_PROMPT_BYTES = 16_384;
export const MAX_AUTHORIZATION_REF = 256;

const CANONICAL_FIELDS = new Set(["query", "response", "reference"]);

export function parsePrompts(text: string): string[] {
  return text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0);
}

export function parseColumnMap(text: string): Record<string, string> | null {
  const trimmed = text.trim();
  if (!trimmed) {
    return {};
  }
  const map: Record<string, string> = {};
  for (const part of trimmed.split(",")) {
    const separator = part.indexOf("=");
    if (separator < 0) {
      return null;
    }
    const field = part.slice(0, separator).trim();
    const column = part.slice(separator + 1).trim();
    if (!field || !column || !CANONICAL_FIELDS.has(field) || field in map) {
      return null;
    }
    map[field] = column;
  }
  return map;
}

export function parseBudget(state: NewRunState): Budget | null {
  const cost = Number(state.budget.cost);
  const seconds = Number(state.budget.seconds);
  const requests = Number(state.budget.requests);
  if (!Number.isFinite(cost) || cost <= 0) {
    return null;
  }
  if (!Number.isFinite(seconds) || seconds <= 0) {
    return null;
  }
  if (!Number.isInteger(requests) || requests <= 0) {
    return null;
  }
  return { max_cost_usd: cost, max_wall_clock_s: seconds, max_target_requests: requests };
}

export function validateNewRun(state: NewRunState): FieldErrors {
  const errors: FieldErrors = {};
  if (state.kind === "paste" && !state.paste.query.trim()) {
    errors.pasteQuery = "请填写问题 / 提示词";
  }
  if (state.kind === "dataset") {
    if (!state.dataset.path.trim()) {
      errors.datasetPath = "请填写数据集文件路径";
    }
    if (parseColumnMap(state.dataset.columnMap) === null) {
      errors.datasetColumnMap = "列映射格式应为 query=q,response=a";
    }
  }
  if (state.kind === "upload" && state.upload.file === null) {
    errors.uploadFile = "请选择上传包";
  }
  if (state.kind === "target") {
    if (!state.target.targetRef) {
      errors.targetRef = "请选择目标";
    }
    const prompts = parsePrompts(state.target.prompts);
    if (prompts.length === 0) {
      errors.targetPrompts = "至少填写一条提示词";
    } else if (prompts.length > MAX_PROMPTS) {
      errors.targetPrompts = "最多 100 条提示词";
    } else if (prompts.some((prompt) => new TextEncoder().encode(prompt).length > MAX_PROMPT_BYTES)) {
      errors.targetPrompts = "单条提示词过长";
    }
  }
  if (parseBudget(state) === null) {
    errors.budget = "预算必须大于 0";
  }
  if (state.authorizationRef.trim().length > MAX_AUTHORIZATION_REF) {
    errors.authorizationRef = "授权引用过长";
  }
  return errors;
}

export function hasErrors(errors: FieldErrors): boolean {
  return Object.keys(errors).length > 0;
}

export function buildRunInput(state: NewRunState): RunInputBody | null {
  if (state.kind === "paste") {
    const query = state.paste.query.trim();
    if (!query) {
      return null;
    }
    const response = state.paste.response.trim();
    const context = state.paste.context.trim();
    return {
      kind: "paste",
      paste: {
        query,
        ...(response ? { response } : {}),
        ...(context ? { context } : {}),
      },
    };
  }
  if (state.kind === "dataset") {
    const path = state.dataset.path.trim();
    const columnMap = parseColumnMap(state.dataset.columnMap);
    if (!path || columnMap === null) {
      return null;
    }
    return {
      kind: "dataset",
      dataset: {
        path,
        format: state.dataset.format,
        ...(Object.keys(columnMap).length > 0 ? { column_map: columnMap } : {}),
      },
    };
  }
  if (state.kind === "target") {
    const prompts = parsePrompts(state.target.prompts);
    if (!state.target.targetRef || prompts.length === 0) {
      return null;
    }
    return {
      kind: "target",
      target_ref: state.target.targetRef,
      target_prompts: prompts,
    };
  }
  return null;
}

export function buildCreateRunBody(
  input: RunInputBody,
  state: NewRunState,
  graders?: GraderSelection,
): CreateRunBody {
  const budget = parseBudget(state);
  if (budget === null) {
    throw new Error("预算无效");
  }
  const authorizationRef = state.authorizationRef.trim();
  const withGraders: RunInputBody & { graders?: GraderSelection } = graders
    ? { ...input, graders }
    : input;
  return {
    input: withGraders,
    budget,
    ...(authorizationRef ? { authorization_ref: authorizationRef } : {}),
  };
}

export interface NewRunApi {
  upload: (file: File) => Promise<{ upload_id: string }>;
  createRun: (body: CreateRunBody) => Promise<{ run_id: string }>;
}

export async function submitNewRun(
  state: NewRunState,
  graders: GraderSelection | undefined,
  client: NewRunApi,
): Promise<{ run_id: string }> {
  let input = buildRunInput(state);
  if (state.kind === "upload") {
    if (state.upload.file === null) {
      throw new Error("请选择上传包");
    }
    const uploaded = await client.upload(state.upload.file);
    input = { kind: "upload", upload_id: uploaded.upload_id };
  }
  if (input === null) {
    throw new Error("输入不完整");
  }
  return client.createRun(buildCreateRunBody(input, state, graders));
}
