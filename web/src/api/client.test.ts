import { afterEach, describe, expect, it, vi } from "vitest";

import { api, setCsrfToken } from "./client";

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
    expect(JSON.parse((fetchMock.mock.calls[0][1] as RequestInit).body as string)).toEqual({
      action: "extend",
      budget: { max_cost_usd: 2, max_wall_clock_s: 90, max_target_requests: 20 },
    });
  });
});
