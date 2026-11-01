import { describe, expect, it, vi } from "vitest";

import type { CreateRunBody } from "../api/types";
import {
  INITIAL_NEW_RUN,
  MAX_AUTHORIZATION_REF,
  buildCreateRunBody,
  buildRunInput,
  parseBudget,
  parseColumnMap,
  parsePrompts,
  submitNewRun,
  validateNewRun,
} from "./newRun";
import type { NewRunApi, NewRunState } from "./newRun";

function state(overrides: Partial<NewRunState> = {}): NewRunState {
  return { ...INITIAL_NEW_RUN, ...overrides };
}

function fakeFile(): File {
  return { name: "evidence.zip" } as unknown as File;
}

const graders = { quality: ["q1"], security: ["s1"] };

describe("parsePrompts", () => {
  it("trims lines and drops blank ones", () => {
    expect(parsePrompts("  a  \n\n b\r\n")).toEqual(["a", "b"]);
  });

  it("returns an empty list for whitespace", () => {
    expect(parsePrompts(" \n\t")).toEqual([]);
  });
});

describe("parseColumnMap", () => {
  it("parses canonical fields", () => {
    expect(parseColumnMap("query=q, response=a")).toEqual({ query: "q", response: "a" });
  });

  it("treats a blank input as an empty map", () => {
    expect(parseColumnMap("   ")).toEqual({});
  });

  it("rejects unknown canonical fields", () => {
    expect(parseColumnMap("prompt=q")).toBeNull();
  });

  it("rejects missing values", () => {
    expect(parseColumnMap("query=")).toBeNull();
  });

  it("rejects duplicate fields", () => {
    expect(parseColumnMap("query=q,query=r")).toBeNull();
  });
});

describe("parseBudget", () => {
  it("converts the three inputs", () => {
    expect(parseBudget(state())).toEqual({
      max_cost_usd: 1,
      max_wall_clock_s: 300,
      max_target_requests: 5,
    });
  });

  it("rejects zero, non-numeric and fractional request counts", () => {
    expect(parseBudget(state({ budget: { cost: "0", seconds: "300", requests: "5" } }))).toBeNull();
    expect(parseBudget(state({ budget: { cost: "1", seconds: "abc", requests: "5" } }))).toBeNull();
    expect(parseBudget(state({ budget: { cost: "1", seconds: "300", requests: "1.5" } }))).toBeNull();
  });
});

describe("validateNewRun", () => {
  it("requires a pasted query", () => {
    expect(validateNewRun(state())).toMatchObject({ pasteQuery: "请填写问题 / 提示词" });
  });

  it("requires a dataset path and accepts an empty column map", () => {
    const errors = validateNewRun(state({ kind: "dataset" }));
    expect(errors.datasetPath).toBe("请填写数据集文件路径");
    expect(errors.datasetColumnMap).toBeUndefined();
  });

  it("rejects an invalid column map", () => {
    const errors = validateNewRun(
      state({ kind: "dataset", dataset: { path: "x.csv", format: "csv", columnMap: "nope=q" } }),
    );
    expect(errors.datasetColumnMap).toBe("列映射格式应为 query=q,response=a");
  });

  it("requires an upload file", () => {
    expect(validateNewRun(state({ kind: "upload" })).uploadFile).toBe("请选择上传包");
    expect(
      validateNewRun(state({ kind: "upload", upload: { file: fakeFile() } })).uploadFile,
    ).toBeUndefined();
  });

  it("requires a target and at least one prompt", () => {
    const errors = validateNewRun(state({ kind: "target" }));
    expect(errors.targetRef).toBe("请选择目标");
    expect(errors.targetPrompts).toBe("至少填写一条提示词");
  });

  it("rejects more than 100 prompts", () => {
    const prompts = Array.from({ length: 101 }, (_, index) => `p${index}`).join("\n");
    expect(
      validateNewRun(state({ kind: "target", target: { targetRef: "demo", prompts } })).targetPrompts,
    ).toBe("最多 100 条提示词");
  });

  it("rejects a prompt over the byte limit", () => {
    const prompts = "字".repeat(6000);
    expect(
      validateNewRun(state({ kind: "target", target: { targetRef: "demo", prompts } })).targetPrompts,
    ).toBe("单条提示词过长");
  });

  it("rejects an invalid budget and an over-long authorization ref", () => {
    const errors = validateNewRun(
      state({
        paste: { query: "q", response: "", context: "" },
        budget: { cost: "0", seconds: "300", requests: "5" },
        authorizationRef: "x".repeat(MAX_AUTHORIZATION_REF + 1),
      }),
    );
    expect(errors.budget).toBe("预算必须大于 0");
    expect(errors.authorizationRef).toBe("授权引用过长");
  });
});

describe("buildRunInput", () => {
  it("omits empty paste fields", () => {
    expect(
      buildRunInput(state({ paste: { query: " q ", response: "", context: " c " } })),
    ).toEqual({ kind: "paste", paste: { query: "q", context: "c" } });
  });

  it("omits an empty column map", () => {
    expect(
      buildRunInput(
        state({ kind: "dataset", dataset: { path: " data.csv ", format: "csv", columnMap: "" } }),
      ),
    ).toEqual({ kind: "dataset", dataset: { path: "data.csv", format: "csv" } });
  });

  it("splits target prompts", () => {
    expect(
      buildRunInput(state({ kind: "target", target: { targetRef: "demo", prompts: "a\n\n b " } })),
    ).toEqual({ kind: "target", target_ref: "demo", target_prompts: ["a", "b"] });
  });

  it("returns null for upload input before the upload step", () => {
    expect(buildRunInput(state({ kind: "upload", upload: { file: fakeFile() } }))).toBeNull();
  });
});

describe("buildCreateRunBody", () => {
  it("adds graders and trims the authorization ref", () => {
    const body = buildCreateRunBody(
      { kind: "paste", paste: { query: "q" } },
      state({ authorizationRef: " case-1 " }),
      graders,
    );
    expect(body).toEqual({
      input: { kind: "paste", paste: { query: "q" }, graders },
      budget: { max_cost_usd: 1, max_wall_clock_s: 300, max_target_requests: 5 },
      authorization_ref: "case-1",
    });
  });

  it("omits empty optional fields", () => {
    const body = buildCreateRunBody({ kind: "paste", paste: { query: "q" } }, state());
    expect(body.input).toEqual({ kind: "paste", paste: { query: "q" } });
    expect("authorization_ref" in body).toBe(false);
  });

  it("throws on an invalid budget", () => {
    expect(() =>
      buildCreateRunBody(
        { kind: "paste", paste: { query: "q" } },
        state({ budget: { cost: "0", seconds: "1", requests: "1" } }),
      ),
    ).toThrow("预算无效");
  });
});

describe("submitNewRun", () => {
  it("uploads first and creates the run with the upload id", async () => {
    const upload = vi.fn().mockResolvedValue({ upload_id: "upload_1" });
    const createRun = vi.fn().mockResolvedValue({ run_id: "run_1" });
    const client: NewRunApi = { upload, createRun };
    const file = fakeFile();

    const result = await submitNewRun(state({ kind: "upload", upload: { file } }), graders, client);

    expect(result).toEqual({ run_id: "run_1" });
    expect(upload).toHaveBeenCalledWith(file);
    const body = createRun.mock.calls[0][0] as CreateRunBody;
    expect(body.input).toEqual({ kind: "upload", upload_id: "upload_1", graders });
  });

  it("does not create a run when the upload fails", async () => {
    const upload = vi.fn().mockRejectedValue(new Error("上传包未通过安全校验"));
    const createRun = vi.fn();
    const client: NewRunApi = { upload, createRun };

    await expect(
      submitNewRun(state({ kind: "upload", upload: { file: fakeFile() } }), undefined, client),
    ).rejects.toThrow("上传包未通过安全校验");
    expect(createRun).not.toHaveBeenCalled();
  });

  it("creates non-upload runs directly", async () => {
    const upload = vi.fn();
    const createRun = vi.fn().mockResolvedValue({ run_id: "run_2" });
    const client: NewRunApi = { upload, createRun };

    await submitNewRun(
      state({ paste: { query: "q", response: "", context: "" } }),
      undefined,
      client,
    );

    expect(upload).not.toHaveBeenCalled();
    const body = createRun.mock.calls[0][0] as CreateRunBody;
    expect(body.input).toEqual({ kind: "paste", paste: { query: "q" } });
  });
});
