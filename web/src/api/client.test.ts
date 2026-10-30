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
});
