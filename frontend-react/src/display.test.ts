import { describe, expect, it } from "vitest";
import { metric, runMetrics } from "./display";
import { safeDownload, search } from "./api/client";
import { createPending } from "./api/submission";
import { mergeEvents } from "./pages/Jobs";
import type { JobEvent } from "./api/contracts";

describe("research display boundaries", () => {
  it("distinguishes missing metrics from a valid zero and preserves rate units", () => {
    expect(metric("total_return", null)).toBe("未定义");
    expect(metric("total_return", 0)).toBe("0%");
    expect(metric("total_return", 0.05)).toBe("5%");
    expect(metric("sharpe", Infinity)).toBe("未定义");
  });
  it("never substitutes RL metrics for an absent baseline", () => {
    expect(
      runMetrics({ metrics: { total_return: 0.1 }, baselines: {} }, "cash"),
    ).toEqual({});
  });
  it("limits downloads to the artifact ID endpoint", () => {
    expect(
      safeDownload(
        "/api/v1/artifacts/12345678-1234-1234-1234-123456789abc/download",
      ),
    ).not.toBeNull();
    expect(safeDownload("https://example.com/model.zip")).toBeNull();
    expect(safeDownload("/api/v1/artifacts/../../secret/download")).toBeNull();
    expect(safeDownload("//example.com/download")).toBeNull();
  });
  it("URL encodes opaque cursors without dropping zero", () => {
    expect(search({ cursor: "a+b/=c", after_seq: 0, absent: null })).toBe(
      "?cursor=a%2Bb%2F%3Dc&after_seq=0",
    );
  });
});

describe("recovery data", () => {
  it("freezes the submitted payload independently from subsequent draft edits", () => {
    const draft = { seeds: [1, 2], trading_config: { commission: 0.001 } };
    const pending = createPending("/experiments", draft);
    draft.seeds.push(3);
    draft.trading_config.commission = 0.02;
    const restored = JSON.parse(JSON.stringify(pending));
    expect(restored.key).toBe(pending.key);
    expect(JSON.parse(restored.payload)).toEqual({
      seeds: [1, 2],
      trading_config: { commission: 0.001 },
    });
    expect(createPending("/experiments", draft).key).not.toBe(pending.key);
  });
  it("merges overlapping cursor pages without losing state transition order", () => {
    const event = (seq: number): JobEvent => ({
      job_id: "job",
      seq,
      occurred_at: "2026-09-08T00:00:00Z",
      event_type: "state",
      payload: { status: seq === 3 ? "succeeded" : "running", revision: seq },
    });
    const merged = mergeEvents([event(1), event(2)], [event(2), event(3)]);
    expect(merged.map((item) => item.seq)).toEqual([1, 2, 3]);
    expect(merged[2].payload).toEqual({ status: "succeeded", revision: 3 });
  });
});
