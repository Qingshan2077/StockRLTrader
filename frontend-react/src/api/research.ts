import { useRef, useState } from "react";
import { ApiError, request } from "./client";
import {
  createPending,
  readStored,
  removeStored,
  saveStored,
  type PendingSubmission,
} from "./submission";
import type { JobDetail, Page } from "./contracts";
import type {
  ResearchProtocolDraft,
  ResearchPreview,
  ResearchProtocol,
  ResearchReport,
} from "./research.contracts";

// Public service projections: large raw bars and private filesystem paths are omitted.
export interface MarketPreview {
  qualification: "ready" | "incomplete" | "unsupported";
  blocking_reasons: string[];
  issues: string[];
  file_hashes: Record<string, string>;
  fingerprint: string;
  rows: number;
  metadata: {
    instrument_id: string;
    exchange: string;
    symbol: string;
    source: string;
    coverage_start: string;
    coverage_end: string;
  } | null;
  market_profile: { profile_id: string } | null;
}
export interface MarketDataset extends MarketPreview {
  dataset_id: string;
  created_at: string;
}
export type Draft = Pick<
  ResearchProtocolDraft,
  | "protocol_id"
  | "hypothesis"
  | "scope"
  | "instrument_ids"
  | "dataset_ids"
  | "dataset_fingerprints"
  | "market_profile_ids"
  | "market_profile_fingerprints"
  | "asset_selection_note"
  | "first_test_session"
  | "fold_count"
  | "seeds"
  | "training_budget"
  | "qualification"
>;
export interface Preview extends ResearchPreview {
  exposure_warnings: string[];
  protocol_sha256: string;
  canonical_request: ResearchProtocolDraft;
  maximum_runtime_seconds: number;
}
export interface ResearchSummary {
  research_id: string;
  job_id: string;
  created_at: string;
  technical_status: string;
  qualification: ResearchProtocolDraft["qualification"];
  hypothesis: string;
  protocol_sha256: string;
}
export interface ResearchDetailData extends ResearchReport {
  research_id: string;
  job_id: string;
  created_at: string;
  protocol: ResearchProtocol;
  protocol_sha256: string;
  preview: Preview;
  job_status: JobDetail;
  attempts: Array<Record<string, unknown>>;
  unit_progress: Array<{
    instrument_id: string;
    fold_id: string;
    seed: number;
    status: string;
    error_json: string | null;
  }>;
}
export interface ResearchSubmission {
  research_id: string;
  job_id: string;
  protocol_sha256?: string;
}
export type ResearchPage = Page<ResearchSummary>;
export const qualificationNames = {
  ready: "可用于研究",
  incomplete: "数据不完整",
  unsupported: "暂不支持",
  exploratory: "探索性研究",
  declared_holdout: "预先声明的留出研究",
};
export const economicNames = {
  insufficient_evidence: "证据不足",
  no_added_value: "未发现额外价值",
  candidate_edge: "有待进一步验证的优势",
};
const reasons: Record<string, string> = {
  EXPOSED_TEST_OVERLAP: "测试区间曾经被查看或评估，本研究按探索性研究处理。",
  SYNTHETIC_DATA: "合成数据仅用于工程验证，不能作为真实市场证据。",
  DATA_CONTRACT_INCOMPLETE: "数据包缺少研究所需信息，请补全后重新登记。",
  CORPORATE_ACTION_COVERAGE_MISSING: "缺少公司行动覆盖声明。",
  MARKET_PROFILE_UNSUPPORTED: "市场规则未完整覆盖，请提供适用的规则文件。",
  planned_units_incomplete_or_duplicate:
    "计划单元缺失或重复，不能形成完整结论。",
  unit_not_completed: "仍有未完成单元。",
  invalid_fingerprint: "产物指纹校验未通过。",
  rule_coverage_incomplete: "市场规则覆盖不完整。",
  requires_five_fixed_seeds: "正式资格需要固定的五个随机种子 42–46。",
  requires_four_complete_nonoverlapping_folds:
    "需要至少四个完整且不重叠的测试窗口。",
  requires_five_assets: "资产集合研究需要至少五个标的。",
  missing_scenario_or_metrics: "成本场景或指标尚不完整。",
  risk_mismatch: "足够多的单元未达到风险可比标准。",
  exploratory: "探索性结果不构成预先留出测试证据。",
};
export const reasonText = (reason: string) =>
  reasons[reason] ??
  (reasons[reason.slice(reason.lastIndexOf(":") + 1)]
    ? `${reason.slice(0, reason.lastIndexOf(":"))}：${reasons[reason.slice(reason.lastIndexOf(":") + 1)]}`
    : reason);
export function datasetFields(datasets: MarketDataset[]) {
  if (
    !datasets.length ||
    datasets.some(
      (d) => d.qualification !== "ready" || !d.metadata || !d.market_profile,
    )
  )
    throw new Error("请选择已登记且可用于研究的数据包。");
  const instrument_ids = datasets.map(
    (d) => `${d.metadata!.exchange}:${d.metadata!.symbol}`,
  );
  if (new Set(instrument_ids).size !== instrument_ids.length)
    throw new Error("每个标的只能选择一个数据包。");
  const profiles: Record<string, string> = {};
  for (const d of datasets) {
    const id = d.market_profile!.profile_id,
      hash = d.file_hashes["market-profile.json"];
    if (
      !/^[0-9a-f]{64}$/.test(d.fingerprint) ||
      !/^[0-9a-f]{64}$/.test(hash ?? "") ||
      (profiles[id] && profiles[id] !== hash)
    )
      throw new Error("数据或市场规则指纹缺失，或同名规则内容冲突。");
    profiles[id] = hash;
  }
  return {
    instrument_ids,
    dataset_ids: datasets.map((d) => d.dataset_id),
    dataset_fingerprints: Object.fromEntries(
      datasets.map((d) => [d.dataset_id, d.fingerprint]),
    ),
    market_profile_ids: Object.keys(profiles),
    market_profile_fingerprints: profiles,
  };
}
export function canQueue(
  draft: Draft | null,
  preview: { request: string; value: Preview } | null,
) {
  return (
    !!draft &&
    !!preview &&
    JSON.stringify(draft) === preview.request &&
    preview.value.blockers.length === 0
  );
}
export function postResearchAttempt(attempt: PendingSubmission) {
  return request<ResearchSubmission>(attempt.path, {
    method: "POST",
    body: attempt.payload,
    headers: { "Idempotency-Key": attempt.key },
  });
}
export function useResearchSubmission(
  storageKey: string,
  onSuccess: (result: ResearchSubmission) => void,
) {
  const [pending, setPending] = useState<PendingSubmission | null>(() =>
    readStored(storageKey),
  );
  const pendingRef = useRef(pending),
    flight = useRef(false);
  const [busy, setBusy] = useState(false),
    [error, setError] = useState<Error | null>(null);
  async function send(attempt: PendingSubmission) {
    if (flight.current) return;
    flight.current = true;
    setBusy(true);
    setError(null);
    try {
      saveStored(storageKey, attempt);
      pendingRef.current = attempt;
      setPending(attempt);
      const result = await postResearchAttempt(attempt);
      removeStored(storageKey);
      pendingRef.current = null;
      setPending(null);
      onSuccess(result);
    } catch (e) {
      setError(e instanceof Error ? e : new Error("提交失败，请重试原请求。"));
    } finally {
      flight.current = false;
      setBusy(false);
    }
  }
  return {
    pending,
    busy,
    error,
    start: (path: string, body: unknown) => {
      if (!pendingRef.current && !flight.current)
        void send(createPending(path, body));
    },
    retry: () => {
      if (pendingRef.current) void send(pendingRef.current);
    },
    rejected:
      error instanceof ApiError && error.status >= 400 && error.status < 500,
    editRejected: () => {
      if (
        !flight.current &&
        error instanceof ApiError &&
        error.status >= 400 &&
        error.status < 500
      ) {
        removeStored(storageKey);
        pendingRef.current = null;
        setPending(null);
        setError(null);
      }
    },
  };
}
