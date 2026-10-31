import { describe, expect, it } from "vitest";

import type { GraderInfo } from "../api/types";

import { allGraderIds, partitionGraders, toGraderSelection, toggleGrader } from "./graders";

const graders: GraderInfo[] = [
  { id: "relevance", mode: "pointwise", tags: ["quality"], cost: "low" },
  { id: "correctness", mode: "pointwise", tags: ["quality", "reference"], cost: "low" },
  { id: "harmfulness", mode: "pointwise", tags: ["security", "safety"], cost: "low" },
  { id: "threat_analysis", mode: "pointwise", tags: ["security"], cost: "low" },
];

describe("grader selection helpers", () => {
  it("partitions the catalog by dimension", () => {
    const groups = partitionGraders(graders);

    expect(groups.quality.map((grader) => grader.id)).toEqual(["relevance", "correctness"]);
    expect(groups.security.map((grader) => grader.id)).toEqual(["harmfulness", "threat_analysis"]);
  });

  it("checks every catalog grader by default", () => {
    expect([...allGraderIds(graders)]).toEqual([
      "relevance",
      "correctness",
      "harmfulness",
      "threat_analysis",
    ]);
  });

  it("toggles one grader without touching the others", () => {
    const selected = allGraderIds(graders);
    const next = toggleGrader(selected, "harmfulness");

    expect(selected.has("harmfulness")).toBe(true);
    expect(next.has("harmfulness")).toBe(false);
    expect(next.has("relevance")).toBe(true);
  });

  it("sends only the checked graders, split by dimension", () => {
    const selected = new Set(["relevance", "threat_analysis"]);

    expect(toGraderSelection(graders, selected)).toEqual({
      quality: ["relevance"],
      security: ["threat_analysis"],
    });
  });
});
