import { describe, expect, it, vi } from "vitest";
import {
  canQueue,
  datasetFields,
  postResearchAttempt,
  type Draft,
  type MarketDataset,
  type Preview,
} from "../api/research";
import { createPending } from "../api/submission";

const dataset: MarketDataset = {
  dataset_id: "registered",
  qualification: "ready",
  fingerprint: "a".repeat(64),
  file_hashes: { "market-profile.json": "b".repeat(64) },
  metadata: {
    exchange: "SSE",
    symbol: "600000",
    instrument_id: "untrusted-alias",
    source: "audited",
    coverage_start: "2010-01-01",
    coverage_end: "2026-01-01",
  },
  market_profile: { profile_id: "audited-profile" },
  created_at: "2026-01-01",
  blocking_reasons: [],
  issues: [],
  rows: 4000,
};
describe("research submit boundaries", () => {
  it("retries the exact HTTP request and key after an uncertain response", async () => {
    const pending = createPending("/researches", { seeds: [42] });
    const fetcher = vi
      .fn()
      .mockRejectedValueOnce(new TypeError("disconnected"))
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ research_id: "same", job_id: "job" }), {
          status: 202,
        }),
      );
    vi.stubGlobal("fetch", fetcher);
    try {
      await expect(postResearchAttempt(pending)).rejects.toThrow("无法确认");
      await expect(postResearchAttempt(pending)).resolves.toEqual({
        research_id: "same",
        job_id: "job",
      });
      for (const [path, options] of fetcher.mock.calls) {
        expect(path).toBe("/api/v1/researches");
        expect(options.body).toBe(pending.payload);
        expect(options.headers.get("Idempotency-Key")).toBe(pending.key);
      }
    } finally {
      vi.unstubAllGlobals();
    }
  });
  it("never enables submit before a successful preview and invalidates any changed input", () => {
    const draft = {
      ...datasetFields([dataset]),
      protocol_id: "p1",
      seeds: [42],
    } as Draft;
    const preview = {
      request: JSON.stringify(draft),
      value: { blockers: [] } as unknown as Preview,
    };
    expect(canQueue(draft, null)).toBe(false);
    expect(canQueue(draft, preview)).toBe(true);
    expect(canQueue({ ...draft, seeds: [43] }, preview)).toBe(false);
    expect(
      canQueue(draft, {
        ...preview,
        value: { ...preview.value, blockers: ["BUDGET_EXCEEDED"] },
      }),
    ).toBe(false);
  });
  it("obtains exact fingerprints from registered data and canonicalizes asset identity", () => {
    expect(datasetFields([dataset])).toEqual({
      instrument_ids: ["SSE:600000"],
      dataset_ids: ["registered"],
      dataset_fingerprints: { registered: "a".repeat(64) },
      market_profile_ids: ["audited-profile"],
      market_profile_fingerprints: { "audited-profile": "b".repeat(64) },
    });
    expect(() =>
      datasetFields([{ ...dataset, qualification: "incomplete" }]),
    ).toThrow("可用于研究");
    expect(() =>
      datasetFields([dataset, { ...dataset, dataset_id: "another" }]),
    ).toThrow("一个数据包");
  });
  it("rejects a ready projection whose input fingerprint is missing", () => {
    expect(() => datasetFields([{ ...dataset, fingerprint: "" }])).toThrow(
      "指纹",
    );
  });
  it("freezes budget and idempotency data for an uncertain submission retry", () => {
    const body = { seeds: [42], training_budget: { requested_timesteps: 100 } };
    const pending = createPending("/researches", body);
    body.training_budget.requested_timesteps = 200;
    expect(
      JSON.parse(pending.payload).training_budget.requested_timesteps,
    ).toBe(100);
    expect(pending.path).toBe("/researches");
    expect(pending.key).toMatch(/^[0-9a-f-]{36}$/);
  });
});
