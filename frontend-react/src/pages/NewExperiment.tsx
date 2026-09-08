import { useEffect, useState, type FormEvent } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { useInfiniteQuery } from "@tanstack/react-query";
import { object, request, search, useCapabilities } from "../api/client";
import type {
  Dataset,
  ExperimentPreview,
  ExperimentRequest,
  Page,
  TradingConfigModel,
} from "../api/contracts";
import {
  readStored,
  removeStored,
  saveStored,
  useSubmission,
} from "../api/submission";
import {
  ConfirmAbandon,
  Empty,
  ErrorBox,
  Loading,
  PageTitle,
} from "../components";
import { number } from "../display";

export const DRAFT_KEY = "stockrl.experiment-draft.v1";
const PENDING_KEY = "stockrl.experiment-pending.v1";
const tradingNames: Record<keyof TradingConfigModel, string> = {
  initial_cash: "初始现金",
  commission: "佣金费率（小数）",
  slippage: "滑点费率（小数）",
  sell_tax: "卖出税率（小数）",
  min_commission: "最低佣金",
  lot_size: "交易手数单位",
  max_participation: "最大成交量参与率（小数）",
  t_plus_one: "启用 T+1",
  drawdown_penalty: "回撤惩罚系数",
  turnover_penalty: "换手惩罚系数",
};
export function defaultsRequest(
  defaults: Record<string, unknown>,
  datasetId: string,
): ExperimentRequest {
  const required = [
    "algorithm",
    "purpose",
    "research_question",
    "comparison_notes",
    "timesteps",
    "seeds",
    "episode_length",
    "train_ratio",
    "val_ratio",
    "trading_config",
    "start_date",
    "end_date",
  ];
  if (
    required.some((key) => !(key in defaults)) ||
    Object.keys(tradingNames).some(
      (key) => !(key in object(defaults.trading_config)),
    )
  )
    throw new Error("服务器未提供完整的默认配置，无法初始化实验。");
  return {
    ...structuredClone(defaults),
    dataset_id: datasetId,
  } as unknown as ExperimentRequest;
}
export default function NewExperiment() {
  const navigate = useNavigate();
  const [params] = useSearchParams();
  const caps = useCapabilities();
  const datasets = useInfiniteQuery({
    queryKey: ["dataset-options"],
    initialPageParam: null as string | null,
    queryFn: ({ pageParam, signal }) =>
      request<Page<Dataset>>(
        `/datasets${search({ cursor: pageParam, limit: 100 })}`,
        { signal },
      ),
    getNextPageParam: (page) => page.next_cursor ?? undefined,
  });
  const [draft, setDraft] = useState<ExperimentRequest | null>(() =>
    readStored<ExperimentRequest>(DRAFT_KEY),
  );
  const [error, setError] = useState<Error | null>(null);
  const [preview, setPreview] = useState<ExperimentPreview | null>(null);
  const [previewBusy, setPreviewBusy] = useState(false);
  const [seedsText, setSeedsText] = useState(
    () => draft?.seeds?.join(", ") ?? "",
  );
  const submission = useSubmission(PENDING_KEY, (result) => {
    removeStored(DRAFT_KEY);
    navigate(`/jobs/${result.job_id}`);
  });
  useEffect(() => {
    if (!draft && caps.data) {
      try {
        const initial = defaultsRequest(
          caps.data.defaults,
          params.get("dataset") ?? "",
        );
        setDraft(initial);
        setSeedsText(initial.seeds.join(", "));
      } catch (reason) {
        setError(reason as Error);
      }
    }
  }, [caps.data, draft, params]);
  useEffect(() => {
    if (draft) {
      try {
        saveStored(DRAFT_KEY, draft);
      } catch {
        setError(
          new Error(
            "浏览器无法保存草稿。请允许会话存储后再提交，以便断线恢复。",
          ),
        );
      }
    }
  }, [draft]);
  const locked = !!submission.pending || submission.busy || previewBusy;
  function update<K extends keyof ExperimentRequest>(
    key: K,
    value: ExperimentRequest[K],
  ) {
    setDraft((old) => (old ? { ...old, [key]: value } : old));
    setPreview(null);
  }
  async function makePreview(event: FormEvent) {
    event.preventDefault();
    if (!draft || locked) return;
    setError(null);
    setPreview(null);
    setPreviewBusy(true);
    try {
      const seedParts = seedsText.split(/[,，\s]+/).filter(Boolean);
      if (!seedParts.length || seedParts.some((item) => !/^\d+$/.test(item)))
        throw new Error("seed 需要是非负整数，用逗号或空格分隔。");
      const seeds = seedParts.map(Number);
      if (new Set(seeds).size !== seeds.length)
        throw new Error("seed 不能重复。");
      const normalized = { ...draft, seeds };
      setDraft(normalized);
      const result = await request<ExperimentPreview>("/experiment-previews", {
        method: "POST",
        body: JSON.stringify(normalized),
      });
      setPreview(result);
    } catch (reason) {
      setError(reason instanceof Error ? reason : new Error("无法预览实验。"));
    } finally {
      setPreviewBusy(false);
    }
  }
  const options = datasets.data?.pages.flatMap((page) => page.items) ?? [];
  const selected = options.find(
    (item) => item.dataset_id === draft?.dataset_id,
  );
  const seedLimits = object(caps.data?.limits.seeds);
  const stepsLimits = object(caps.data?.limits.timesteps);
  return (
    <>
      <PageTitle eyebrow="EXPERIMENT / 研究配置" title="创建实验">
        <p className="hint">
          配置草稿保存在当前浏览器会话；预览确认后才进入队列。
        </p>
      </PageTitle>
      {!caps.data && caps.isPending && <Loading />}
      <ErrorBox error={caps.error} retry={() => void caps.refetch()} />
      <ErrorBox error={datasets.error} retry={() => void datasets.refetch()} />
      <ErrorBox error={error} />
      {!submission.pending && <ErrorBox error={submission.error} />}
      {params.get("dataset") &&
        draft &&
        draft.dataset_id !== params.get("dataset") &&
        !locked && (
          <div className="notice warning">
            <p>已恢复之前的配置草稿。你刚选择的数据集尚未替换草稿中的数据。</p>
            <button
              onClick={() => update("dataset_id", params.get("dataset")!)}
            >
              将草稿改用刚选择的数据集
            </button>
          </div>
        )}
      {submission.pending && (
        <section className="panel">
          <h2>有一笔提交等待确认</h2>
          <p>
            继续使用原请求和同一提交编号查询或重试。刷新页面不会创建新任务。
          </p>
          <p className="mono break">{submission.pending.key}</p>
          <details>
            <summary>查看已锁定的提交配置</summary>
            <pre>
              {JSON.stringify(JSON.parse(submission.pending.payload), null, 2)}
            </pre>
          </details>
          <ErrorBox error={submission.error} />
          <div className="actions">
            <button disabled={submission.busy} onClick={submission.retry}>
              {submission.busy ? "正在确认…" : "确认 / 重试原提交"}
            </button>
            <ConfirmAbandon onConfirm={submission.abandon} />
          </div>
        </section>
      )}
      {draft && caps.data && (
        <form onSubmit={(event) => void makePreview(event)}>
          <fieldset disabled={locked}>
            <section className="panel">
              <h2>研究问题与数据</h2>
              <div className="form-grid">
                <label className="wide">
                  数据快照
                  <select
                    required
                    value={draft.dataset_id}
                    onChange={(event) =>
                      update("dataset_id", event.target.value)
                    }
                  >
                    <option value="">选择已保存的数据集</option>
                    {draft.dataset_id && !selected && (
                      <option value={draft.dataset_id}>
                        {draft.dataset_id}（未在当前列表）
                      </option>
                    )}
                    {options.map((item) => (
                      <option key={item.dataset_id} value={item.dataset_id}>
                        {item.display_name}
                        {item.is_synthetic ? " · 合成演示" : ""} · {item.rows}{" "}
                        行
                      </option>
                    ))}
                  </select>
                </label>
                {datasets.hasNextPage && (
                  <button
                    type="button"
                    className="secondary"
                    disabled={datasets.isFetchingNextPage}
                    onClick={() => void datasets.fetchNextPage()}
                  >
                    加载更多数据集
                  </button>
                )}
                {datasets.data && !options.length && (
                  <Empty>请先到数据集页面导入数据。</Empty>
                )}
                <label>
                  实验用途
                  <select
                    value={draft.purpose}
                    onChange={(event) =>
                      update(
                        "purpose",
                        event.target.value as ExperimentRequest["purpose"],
                      )
                    }
                  >
                    <option value="technical_validation">技术验证</option>
                    <option value="research">研究实验</option>
                  </select>
                </label>
                <label>
                  算法
                  <select
                    value={draft.algorithm}
                    onChange={(event) =>
                      update(
                        "algorithm",
                        event.target.value as ExperimentRequest["algorithm"],
                      )
                    }
                  >
                    {caps.data.algorithms.map((name) => (
                      <option key={name}>{name}</option>
                    ))}
                  </select>
                </label>
                <label className="wide">
                  研究问题
                  <textarea
                    value={draft.research_question}
                    required={draft.purpose === "research"}
                    maxLength={10000}
                    rows={3}
                    onChange={(event) =>
                      update("research_question", event.target.value)
                    }
                    placeholder="说明要检验的假设、对照与可能推翻假设的结果"
                  />
                </label>
                <label className="wide">
                  比较说明
                  <textarea
                    value={draft.comparison_notes}
                    maxLength={10000}
                    rows={2}
                    onChange={(event) =>
                      update("comparison_notes", event.target.value)
                    }
                  />
                </label>
                <label>
                  开始日期（含）
                  <input
                    type="date"
                    value={draft.start_date ?? ""}
                    onChange={(event) =>
                      update("start_date", event.target.value || null)
                    }
                  />
                </label>
                <label>
                  结束日期（含）
                  <input
                    type="date"
                    value={draft.end_date ?? ""}
                    onChange={(event) =>
                      update("end_date", event.target.value || null)
                    }
                  />
                </label>
              </div>
              {selected?.is_synthetic && (
                <p className="notice warning">
                  该快照为合成演示数据。结果仅能说明流程表现。
                </p>
              )}
            </section>
            <section className="panel">
              <h2>训练与日期切分</h2>
              <div className="form-grid">
                <label>
                  每个 seed 的计划训练步数
                  <input
                    type="number"
                    required
                    min={Number(stepsLimits.min)}
                    max={Number(stepsLimits.max)}
                    step={1}
                    value={draft.timesteps}
                    onChange={(event) =>
                      update("timesteps", Number(event.target.value))
                    }
                  />
                </label>
                <label>
                  随机种子
                  <input
                    required
                    value={seedsText}
                    onChange={(event) => {
                      setSeedsText(event.target.value);
                      setPreview(null);
                      const tokens = event.target.value
                        .split(/[,，\s]+/)
                        .filter(Boolean);
                      if (
                        tokens.length &&
                        tokens.every((token) => /^\d+$/.test(token))
                      )
                        update("seeds", tokens.map(Number));
                    }}
                  />
                  <span className="hint">
                    最多 {String(seedLimits.max_count)}{" "}
                    个，全部展示，不按测试成绩挑选。
                  </span>
                </label>
                <label>
                  训练片段长度
                  <input
                    type="number"
                    min={1}
                    step={1}
                    value={draft.episode_length ?? ""}
                    onChange={(event) =>
                      update(
                        "episode_length",
                        event.target.value ? Number(event.target.value) : null,
                      )
                    }
                  />
                  <span className="hint">留空使用完整训练区间。</span>
                </label>
                <label>
                  训练集比例
                  <input
                    type="number"
                    min={0}
                    max={1}
                    step="any"
                    required
                    value={draft.train_ratio}
                    onChange={(event) =>
                      update("train_ratio", Number(event.target.value))
                    }
                  />
                </label>
                <label>
                  验证集比例
                  <input
                    type="number"
                    min={0}
                    max={1}
                    step="any"
                    required
                    value={draft.val_ratio}
                    onChange={(event) =>
                      update("val_ratio", Number(event.target.value))
                    }
                  />
                </label>
                <div>
                  <span className="field-label">测试集比例</span>
                  <p className="numeric">
                    {number((1 - draft.train_ratio - draft.val_ratio) * 100, 2)}
                    %
                  </p>
                </div>
              </div>
              <p className="hint">
                归一化只拟合训练集；验证集选择检查点；测试集用于最终报告。
              </p>
            </section>
            <section className="panel">
              <h2>成交约束与奖励</h2>
              <div className="form-grid">
                {(
                  Object.keys(tradingNames) as Array<keyof TradingConfigModel>
                ).map((key) => {
                  const value = draft.trading_config[key];
                  const bounds = object(
                    object(caps.data?.limits.trading_config)[key],
                  );
                  return (
                    <label
                      key={key}
                      className={
                        typeof value === "boolean" ? "check-label" : ""
                      }
                    >
                      {tradingNames[key]}
                      {typeof value === "boolean" ? (
                        <input
                          type="checkbox"
                          checked={value}
                          onChange={(event) =>
                            update("trading_config", {
                              ...draft.trading_config,
                              [key]: event.target.checked,
                            })
                          }
                        />
                      ) : (
                        <input
                          type="number"
                          required
                          step={bounds.type === "integer" ? 1 : "any"}
                          min={
                            typeof bounds.minimum === "number"
                              ? bounds.minimum
                              : undefined
                          }
                          max={
                            typeof bounds.maximum === "number"
                              ? bounds.maximum
                              : undefined
                          }
                          value={value}
                          onChange={(event) =>
                            update("trading_config", {
                              ...draft.trading_config,
                              [key]: Number(event.target.value),
                            })
                          }
                        />
                      )}
                    </label>
                  );
                })}
              </div>
              <p className="hint">
                费率使用小数，例如 0.001 表示
                0.1%。现金和费用使用数据集声明的计价单位。
              </p>
            </section>
            <button type="submit">
              {previewBusy ? "正在核对切分…" : "预览日期切分与配置"}
            </button>
          </fieldset>
        </form>
      )}
      {preview && !submission.pending && (
        <section className="panel preview">
          <h2>确认实验范围</h2>
          <div className="split-band">
            {Object.entries(preview.splits).map(([key, interval]) => (
              <div key={key}>
                <strong>
                  {{ train: "训练", validation: "验证", test: "测试" }[key] ??
                    key}
                </strong>
                <span>观察起点 {interval.observation_start_date}</span>
                <span>
                  收益 {interval.first_reward_date} —{" "}
                  {interval.last_reward_date}
                </span>
                <b>{interval.reward_count} 个收益区间</b>
              </div>
            ))}
          </div>
          {preview.warnings.map((warning) => (
            <p key={warning} className="notice warning">
              {warning}
            </p>
          ))}
          <p>
            {preview.seed_count} 个 seed · {preview.canonical_request.algorithm}{" "}
            · 每个 seed {number(preview.canonical_request.timesteps, 0)} 计划步
          </p>
          <p className="hint">
            {caps.data?.worker_available
              ? "确认后加入执行队列。"
              : "执行器当前离线，确认后将排队等待恢复。"}
          </p>
          <details>
            <summary>数据与请求指纹</summary>
            <p className="mono break">数据 {preview.filtered_data_sha256}</p>
            <p className="mono break">请求 {preview.request_sha256}</p>
          </details>
          <button
            disabled={submission.busy}
            onClick={() =>
              submission.start("/experiments", preview.canonical_request)
            }
          >
            确认并加入队列
          </button>
        </section>
      )}
    </>
  );
}
