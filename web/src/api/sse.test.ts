import { describe, expect, it } from "vitest";

import { mergeEvent, parseEventData } from "./sse";

describe("sse helpers", () => {
  it("parses a data frame", () => {
    expect(parseEventData('{"seq":2,"type":"run.created","payload":{"kind":"paste"}}')).toEqual({
      seq: 2,
      type: "run.created",
      payload: { kind: "paste" },
    });
  });

  it("rejects malformed frames", () => {
    expect(parseEventData("not json")).toBeNull();
    expect(parseEventData('{"seq":"2"}')).toBeNull();
  });

  it("merges without duplicates and keeps order", () => {
    const first = { seq: 2, type: "b", payload: {} };
    const second = { seq: 1, type: "a", payload: {} };

    const merged = mergeEvent(mergeEvent([], first), second);

    expect(merged.map((event) => event.seq)).toEqual([1, 2]);
    expect(mergeEvent(merged, first)).toHaveLength(2);
  });
});
