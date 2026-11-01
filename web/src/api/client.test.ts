import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError, api, setCsrfToken } from "./client";

afterEach(() => {
  vi.unstubAllGlobals();
  setCsrfToken(null);
});

describe("api client", () => {
  it("sends the CSRF header on writes", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ run_id: "run_1", status: "queued" }), { status: 201 }),
    );
    vi.stubGlobal("fetch", fetchMock);
    setCsrfToken("token");

    await api.createRun({
      input: { kind: "paste", paste: { query: "q", response: "a" } },
      budget: { max_cost_usd: 1, max_wall_clock_s: 60, max_target_requests: 1 },
    });

    const init = fetchMock.mock.calls[0][1] as RequestInit;
    expect(new Headers(init.headers).get("X-CSRF-Token")).toBe("token");
  });

  it("does not send the CSRF header on reads", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ items: [] }), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);
    setCsrfToken("token");

    await api.listRuns();

    const init = fetchMock.mock.calls[0][1] as RequestInit;
    expect(new Headers(init.headers).get("X-CSRF-Token")).toBeNull();
  });

  it("maps the error envelope", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ error: { code: "invalid_state", message: "nope" } }), { status: 409 }),
      ),
    );

    await expect(api.cancel("run_1")).rejects.toMatchObject({ status: 409, code: "invalid_state" });
  });

  it("uses a Chinese fallback when the error envelope has no message", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response("<html>bad gateway</html>", { status: 502 })),
    );

    const error = (await api
      .budgetDecision("run_1", { action: "summarize" })
      .catch((caught) => caught)) as ApiError;

    expect(error).toBeInstanceOf(ApiError);
    expect(error.status).toBe(502);
    expect(error.code).toBe("http_error");
    expect(error.message).toBe("请求失败，请重试");
    expect(error.message).not.toContain("request failed");
  });

  it("preserves error details", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({
            error: {
              code: "budget_overrun",
              message: "confirm",
              details: { finalize_estimate: 0.2, remaining: 0.1 },
            },
          }),
          { status: 409 },
        ),
      ),
    );

    await expect(
      api.budgetDecision("run_1", { action: "summarize" }),
    ).rejects.toMatchObject({
      code: "budget_overrun",
      details: { finalize_estimate: 0.2, remaining: 0.1 },
    });
  });

  it("posts to the grader package endpoint and returns the metadata", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({ package_id: "pkg_1", run_id: "run-1", sha256: "a", size_bytes: 10, replayable: true }),
        { status: 201, headers: { "Content-Type": "application/json" } },
      ),
    );
    vi.stubGlobal("fetch", fetchMock);

    const result = await api.exportGraderPackage("run-1");

    const [path, init] = fetchMock.mock.calls[0];
    expect(path).toBe("/api/runs/run-1/grader-package");
    expect(init.method).toBe("POST");
    expect(result.package_id).toBe("pkg_1");
  });

  it("reads the stored grader package metadata", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ package_id: "pkg_1", run_id: "run-1", sha256: "a", size_bytes: 10 }), {
        status: 200,
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const result = await api.graderPackage("run-1");

    expect(fetchMock.mock.calls[0][0]).toBe("/api/runs/run-1/grader-package");
    expect(result.package_id).toBe("pkg_1");
  });

  it("builds a same-origin download path without credentials", () => {
    expect(api.graderPackageUrl("run-1")).toBe("/api/runs/run-1/grader-package/download");
    expect(api.graderPackageUrl("run/with space")).toBe(
      "/api/runs/run%2Fwith%20space/grader-package/download",
    );
  });

  it("downloads the grader package through the authenticated client", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response("PK\u0003\u0004", { status: 200, headers: { "Content-Type": "application/zip" } }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const blob = await api.downloadGraderPackage("run-1");

    const [path, init] = fetchMock.mock.calls[0];
    expect(path).toBe("/api/runs/run-1/grader-package/download");
    expect(init).toMatchObject({ credentials: "same-origin" });
    expect(await blob.text()).toBe("PK\u0003\u0004");
  });

  it("surfaces a failed grader package download", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ error: { code: "not_found", message: "评测包不存在" } }), {
          status: 404,
        }),
      ),
    );

    await expect(api.downloadGraderPackage("run-1")).rejects.toMatchObject({
      status: 404,
      code: "not_found",
      message: "评测包不存在",
    });
  });

  it("posts a budget decision to the run route", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ run_id: "run_1", status: "waiting_budget" }), { status: 202 }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await api.budgetDecision("run_1", {
      action: "extend",
      budget: { max_cost_usd: 2, max_wall_clock_s: 90, max_target_requests: 20 },
    });

    expect(fetchMock.mock.calls[0][0]).toBe("/api/runs/run_1/budget/decision");
    expect((fetchMock.mock.calls[0][1] as RequestInit).method).toBe("POST");
    expect(JSON.parse((fetchMock.mock.calls[0][1] as RequestInit).body as string)).toEqual({
      action: "extend",
      budget: { max_cost_usd: 2, max_wall_clock_s: 90, max_target_requests: 20 },
    });
  });

  it("uploads a file as multipart without a JSON content type", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ upload_id: "upload_1", status: "extracted" }), {
        status: 202,
        headers: { "Content-Type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);
    setCsrfToken("token");

    const file = new File(["PK"], "evidence.zip", { type: "application/zip" });
    const result = await api.upload(file);

    const [path, init] = fetchMock.mock.calls[0];
    const headers = new Headers(init.headers);
    expect(path).toBe("/api/uploads");
    expect(init.method).toBe("POST");
    expect(init.body).toBeInstanceOf(FormData);
    expect(headers.get("Content-Type")).toBeNull();
    expect(headers.get("X-CSRF-Token")).toBe("token");
    expect(result).toEqual({ upload_id: "upload_1", status: "extracted" });
  });

  it("surfaces an upload rejection as an ApiError", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({ error: { code: "upload_rejected", message: "上传包未通过安全校验" } }),
          { status: 400 },
        ),
      ),
    );

    await expect(api.upload(new File(["PK"], "x.zip"))).rejects.toMatchObject({
      status: 400,
      code: "upload_rejected",
      message: "上传包未通过安全校验",
    });
  });
});
