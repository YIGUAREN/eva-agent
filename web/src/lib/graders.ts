import type { GraderInfo } from "../api/types";

export interface GraderGroups {
  quality: GraderInfo[];
  security: GraderInfo[];
}

/** Split the catalog into the two dimension groups the run input accepts. */
export function partitionGraders(graders: GraderInfo[]): GraderGroups {
  const groups: GraderGroups = { quality: [], security: [] };
  for (const grader of graders) {
    if (grader.tags.includes("quality")) {
      groups.quality.push(grader);
    } else if (grader.tags.includes("security")) {
      groups.security.push(grader);
    }
  }
  return groups;
}

/** Every catalog grader id, the default checked state before the user changes it. */
export function allGraderIds(graders: GraderInfo[]): Set<string> {
  return new Set(graders.map((grader) => grader.id));
}

export function toggleGrader(selected: ReadonlySet<string>, id: string): Set<string> {
  const next = new Set(selected);
  if (next.has(id)) {
    next.delete(id);
  } else {
    next.add(id);
  }
  return next;
}

/** The checked ids, split by dimension; unchecked graders never reach the run. */
export function toGraderSelection(
  graders: GraderInfo[],
  selected: ReadonlySet<string>,
): { quality: string[]; security: string[] } {
  const groups = partitionGraders(graders);
  return {
    quality: groups.quality.filter((grader) => selected.has(grader.id)).map((grader) => grader.id),
    security: groups.security.filter((grader) => selected.has(grader.id)).map((grader) => grader.id),
  };
}
