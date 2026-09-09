import { describe, expect, it } from "vitest";
import {
  researchStates,
  diagnosticValue,
  canResumeResearch,
} from "./ResearchDetail";

describe("research conclusions and recovery", () => {
  it("keeps technical completion independent from economic evidence", () => {
    expect(
      researchStates("completed", "exploratory", "candidate_edge"),
    ).toEqual(["执行完成", "探索性研究", "有待进一步验证的优势"]);
    expect(
      researchStates("completed", "declared_holdout", "no_added_value")[2],
    ).toBe("未发现额外价值");
    expect(
      researchStates("failed", "declared_holdout", "insufficient_evidence")[2],
    ).toBe("证据不足");
  });
  it("never displays missing raw actions as zero clipping", () => {
    expect(diagnosticValue(null)).toBe("未记录 / 无法计算");
    expect(diagnosticValue(0)).toBe("0.0%");
  });
  it("allows explicit resume only after an unsuccessful terminal job", () => {
    for (const state of ["failed", "cancelled", "interrupted"])
      expect(canResumeResearch(state)).toBe(true);
    for (const state of ["queued", "running", "cancelling", "succeeded"])
      expect(canResumeResearch(state)).toBe(false);
  });
});
