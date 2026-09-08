import { lazy, Suspense, useEffect, useState } from "react";
import {
  Link,
  useNavigate,
  useParams,
  useSearchParams,
} from "react-router-dom";
import { useQueries, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  downloadArtifact,
  object,
  request,
  safeDownload,
  search,
  terminal,
  useJob,
} from "../api/client";
import type {
  Dataset,
  ExperimentDetail as Experiment,
  Page,
  Policy,
  SeriesResponse,
  TradesResponse,
} from "../api/contracts";
import { saveStored, useSubmission } from "../api/submission";
import {
  Badge,
  ConfirmAbandon,
  CursorButtons,
  DatasetFacts,
  Empty,
  ErrorBox,
  ExperimentLink,
  JobBadge,
  JobProgress,
  Loading,
  PageTitle,
  Provenance,
  useCursor,
} from "../components";
import {
  algorithm,
  experimentName,
  metric,
  metrics,
  number,
  policyNames,
  runMetrics,
  statusNames,
  timestamp,
} from "../display";
import { DRAFT_KEY } from "./NewExperiment";

const Charts = lazy(() => import("../Charts"));
function ActualStatus({ id }: { id: string | null }) {
  const job = useJob(id);
  const client = useQueryClient();
  useEffect(() => {
    if (terminal(job.data?.status))
      void client.invalidateQueries({ queryKey: ["experiments"] });
  }, [job.data?.status, client]);
  return !id ? (
    <Badge>历史产物</Badge>
  ) : job.data ? (
    <JobBadge job={job.data} />
  ) : (
    <span className="hint">{job.error ? "状态待确认" : "读取状态…"}</span>
  );
}
export function Experiments() {
  const cursor = useCursor();
  const [filters, setFilters] = useState({
    status: "",
    kind: "",
    algorithm: "",
    dataset_id: "",
  });
  const [selection, setSelection] = useState<string[]>([]);
  const query = useQuery({
    queryKey: ["experiments", filters, cursor.cursor],
    queryFn: ({ signal }) =>
      request<Page<Experiment>>(
        `/experiments${search({ ...filters, cursor: cursor.cursor })}`,
        { signal },
      ),
  });
  const datasets = useQuery({
    queryKey: ["datasets-filter"],
    queryFn: ({ signal }) =>
      request<Page<Dataset>>("/datasets?limit=100", { signal }),
  });
  const navigate = useNavigate();
  function filter(key: keyof typeof filters, value: string) {
    setFilters((old) => ({ ...old, [key]: value }));
    cursor.reset();
  }
  return (
    <>
      <PageTitle eyebrow="ARCHIVE / 实验记录" title="实验">
        <div className="actions">
          <button
            disabled={!selection.length}
            onClick={() =>
              navigate(
                `/compare?${selection.map((id) => `ids=${id}`).join("&")}`,
              )
            }
          >
            比较已选（{selection.length}/4）
          </button>
          <Link className="button secondary" to="/experiments/new">
            创建实验
          </Link>
        </div>
      </PageTitle>
      <div className="filters">
        <label>
          任务状态
          <select
            value={filters.status}
            onChange={(event) => filter("status", event.target.value)}
          >
            <option value="">全部状态</option>
            {Object.entries(statusNames).map(([key, value]) => (
              <option key={key} value={key}>
                {value}
              </option>
            ))}
          </select>
        </label>
        <label>
          实验类型
          <select
            value={filters.kind}
            onChange={(event) => filter("kind", event.target.value)}
          >
            <option value="">全部类型</option>
            <option value="train">训练</option>
            <option value="replay">重放</option>
          </select>
        </label>
        <label>
          算法
          <select
            value={filters.algorithm}
            onChange={(event) => filter("algorithm", event.target.value)}
          >
            <option value="">全部算法</option>
            <option>PPO</option>
            <option>SAC</option>
          </select>
        </label>
        <label>
          数据集
          <select
            value={filters.dataset_id}
            onChange={(event) => filter("dataset_id", event.target.value)}
          >
            <option value="">全部数据集</option>
            {datasets.data?.items.map((item) => (
              <option key={item.dataset_id} value={item.dataset_id}>
                {item.display_name}
              </option>
            ))}
          </select>
        </label>
      </div>
      <ErrorBox error={query.error} retry={() => void query.refetch()} />
      {query.isPending && <Loading />}
      {query.data && !query.data.items.length && (
        <Empty>没有符合条件的实验。可调整筛选或创建实验。</Empty>
      )}
      {query.data && (
        <div className="panel table-scroll">
          <table>
            <thead>
              <tr>
                <th>比较</th>
                <th>数据与实验</th>
                <th>算法 / seed</th>
                <th>任务状态</th>
                <th>用途与产物</th>
                <th>创建时间</th>
              </tr>
            </thead>
            <tbody>
              {query.data.items.map((item) => (
                <tr key={item.experiment_id}>
                  <td>
                    <input
                      type="checkbox"
                      aria-label={`比较 ${experimentName(item)} ${item.experiment_id.slice(0, 8)}`}
                      checked={selection.includes(item.experiment_id)}
                      disabled={
                        item.kind !== "train" ||
                        item.integrity !== "complete" ||
                        (selection.length >= 4 &&
                          !selection.includes(item.experiment_id))
                      }
                      onChange={(event) =>
                        setSelection((old) =>
                          event.target.checked
                            ? [...old, item.experiment_id]
                            : old.filter((id) => id !== item.experiment_id),
                        )
                      }
                    />
                  </td>
                  <td>
                    <ExperimentLink item={item} />
                  </td>
                  <td>
                    {algorithm(item)}
                    <small className="mono">
                      {(
                        item.request?.seeds ??
                        item.legacy_config?.seeds ??
                        []
                      ).join(", ") || "—"}
                    </small>
                  </td>
                  <td>
                    <ActualStatus id={item.job_id} />
                  </td>
                  <td>
                    <Provenance experiment={item} />
                  </td>
                  <td>{timestamp(item.created_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <CursorButtons
        hasMore={query.data?.has_more ?? false}
        next={() => cursor.next(query.data?.next_cursor ?? null)}
        previous={cursor.previous}
        canPrevious={cursor.canPrevious}
        busy={query.isFetching}
      />
      <p className="hint">
        比较仅接受完整训练实验；历史产物与任务状态分别标注，不使用产物完整性推断任务成功。
      </p>
    </>
  );
}
export function MetricsTable({
  experiment,
  seed,
}: {
  experiment: Experiment;
  seed?: number;
}) {
  const runs =
    seed === undefined
      ? experiment.runs
      : experiment.runs.filter((run) => run.seed === seed);
  return (
    <div className="table-scroll">
      <table>
        <caption>
          保存的测试指标 · 252 日年化 · 零无风险利率 · null 保留为“未定义”
          <span>
            {" "}
            · 费用单位：
            {experiment.dataset && experiment.dataset.quote_unit !== "unknown"
              ? experiment.dataset.quote_unit
              : "未声明计价单位"}
          </span>
        </caption>
        <thead>
          <tr>
            <th>seed / 策略</th>
            {Object.entries(metrics).map(([key, value]) => (
              <th key={key}>{value.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {runs.flatMap((run) =>
            (Object.keys(policyNames) as Policy[]).map((policy) => (
              <tr key={`${String(run.seed)}-${policy}`}>
                <th className="row-label">
                  {String(run.seed)} · {policyNames[policy]}
                </th>
                {Object.keys(metrics).map((key) => (
                  <td className="numeric" key={key}>
                    {metric(key, runMetrics(run, policy)[key])}
                  </td>
                ))}
              </tr>
            )),
          )}
        </tbody>
      </table>
    </div>
  );
}
export function AggregateTable({ experiment }: { experiment: Experiment }) {
  const baseline = object(experiment.baseline_aggregate);
  return (
    <div className="table-scroll">
      <table>
        <caption>
          全部 seed 汇总 · 均值 / 总体标准差 / 有效样本数（ddof=0）
          <span>
            {" "}
            · 费用单位：
            {experiment.dataset && experiment.dataset.quote_unit !== "unknown"
              ? experiment.dataset.quote_unit
              : "未声明计价单位"}
          </span>
        </caption>
        <thead>
          <tr>
            <th>策略 / 指标</th>
            {Object.entries(metrics).map(([key, value]) => (
              <th key={key}>{value.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {(Object.keys(policyNames) as Policy[]).map((policy) => (
            <tr key={policy}>
              <th className="row-label">{policyNames[policy]}</th>
              {Object.keys(metrics).map((key) => {
                const stat = object(
                  (policy === "rl"
                    ? object(experiment.aggregate)
                    : object(baseline[policy]))[key],
                );
                return (
                  <td className="numeric" key={key}>
                    {metric(key, stat.mean)}
                    <small>
                      σ {metric(key, stat.std)} · n {number(stat.count, 0)}
                    </small>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
function Trades({
  id,
  seed,
  policy,
}: {
  id: string;
  seed: number;
  policy: Policy;
}) {
  const cursor = useCursor();
  const query = useQuery({
    queryKey: ["trades", id, seed, policy, cursor.cursor],
    queryFn: ({ signal }) =>
      request<TradesResponse>(
        `/experiments/${id}/trades${search({ seed, policy, cursor: cursor.cursor })}`,
        { signal },
      ),
  });
  return (
    <section className="panel">
      <h2>成交记录 · {policyNames[policy]}</h2>
      <ErrorBox error={query.error} retry={() => void query.refetch()} />
      {query.isPending && <Loading />}
      {query.data && (
        <>
          <p className="hint">
            共 {query.data.total_rows} 条成交，完整 CSV 在产物列表下载。
          </p>
          {!query.data.items.length ? (
            <Empty>这个 seed 与策略没有成交记录。</Empty>
          ) : (
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    {["日期", "方向", "股数", "成交价格", "费用"].map(
                      (label) => (
                        <th key={label}>{label}</th>
                      ),
                    )}
                  </tr>
                </thead>
                <tbody>
                  {query.data.items.map((item, index) => (
                    <tr key={`${item.date}-${index}`}>
                      <td>{item.date}</td>
                      <td>{item.side === "buy" ? "买入" : "卖出"}</td>
                      <td className="numeric">{number(item.shares)}</td>
                      <td className="numeric">{number(item.price)}</td>
                      <td className="numeric">{number(item.cost)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          <CursorButtons
            hasMore={query.data.has_more}
            next={() => cursor.next(query.data?.next_cursor ?? null)}
            previous={cursor.previous}
            canPrevious={cursor.canPrevious}
            busy={query.isFetching}
          />
        </>
      )}
    </section>
  );
}
function ResultViews({ experiment }: { experiment: Experiment }) {
  const availableSeeds = experiment.runs
    .map((run) => run.seed)
    .filter((seed): seed is number => typeof seed === "number");
  const [seed, setSeed] = useState(availableSeeds[0] ?? null);
  const [policies, setPolicies] = useState<Policy[]>(["rl", "buy_hold"]);
  const [tradePolicy, setTradePolicy] = useState<Policy>("rl");
  const [field, setField] = useState<"nav" | "weights" | "cash" | "cost">(
    "nav",
  );
  const queries = useQueries({
    queries: policies.map((policy) => ({
      queryKey: ["series", experiment.experiment_id, seed, policy],
      queryFn: ({ signal }: { signal: AbortSignal }) =>
        request<SeriesResponse>(
          `/experiments/${experiment.experiment_id}/series${search({ seed, policy, max_points: 5000 })}`,
          { signal },
        ),
      enabled: seed !== null,
    })),
  });
  const unit =
    !experiment.dataset || experiment.dataset.quote_unit === "unknown"
      ? "未声明计价单位"
      : experiment.dataset.quote_unit;
  if (seed === null) return <Empty>当前产物没有可展示的 seed 结果。</Empty>;
  return (
    <>
      <section className="panel">
        <div className="section-heading">
          <h2>策略轨迹</h2>
          <label>
            查看 seed
            <select
              value={seed}
              onChange={(event) => setSeed(Number(event.target.value))}
            >
              {availableSeeds.map((value) => (
                <option key={value}>{value}</option>
              ))}
            </select>
          </label>
        </div>
        <div className="decision-line">
          <span>t 日收盘观察</span>
          <b>→</b>
          <span>t+1 开盘按约束成交</span>
          <b>→</b>
          <span>t+1 收盘估值</span>
        </div>
        <div className="chart-controls">
          <div className="checks">
            {(Object.keys(policyNames) as Policy[]).map((policy) => (
              <label className="check-label" key={policy}>
                <input
                  type="checkbox"
                  checked={policies.includes(policy)}
                  onChange={(event) =>
                    setPolicies((old) =>
                      event.target.checked
                        ? [...old, policy]
                        : old.filter((value) => value !== policy),
                    )
                  }
                />
                {policyNames[policy]}
              </label>
            ))}
          </div>
          <label>
            图表内容
            <select
              value={field}
              onChange={(event) => setField(event.target.value as typeof field)}
            >
              <option value="nav">净值</option>
              <option value="weights">实际 / 请求 / 成交仓位</option>
              <option value="cash">现金</option>
              <option value="cost">单次费用</option>
            </select>
          </label>
        </div>
        {queries.some((q) => q.isPending) && <Loading />}
        {queries.map(
          (query, index) =>
            query.error && (
              <ErrorBox
                key={policies[index]}
                error={query.error}
                retry={() => void query.refetch()}
              />
            ),
        )}
        {!policies.length ? (
          <Empty>选择至少一个策略查看曲线。</Empty>
        ) : (
          <Suspense fallback={<Loading />}>
            <Charts
              series={queries.flatMap((query) =>
                query.data ? [query.data] : [],
              )}
              field={field}
              unit={unit}
            />
          </Suspense>
        )}
        {queries
          .flatMap((query) => (query.data?.is_sampled ? [query.data] : []))
          .map((item) => (
            <p className="notice warning" key={item.policy}>
              {policyNames[item.policy]}：显示抽样 {item.returned_count} /{" "}
              {item.original_count} 个点；指标仍来自完整数据，完整序列可下载。
            </p>
          ))}
        <p className="hint">
          仓位线含义：实际仓位为收盘估值后的比例，请求仓位为决策目标，成交仓位为开盘成交后的比例。现金、净值、费用单位：
          {unit}。
        </p>
        <p className="hint">
          “买入并持有”基准持续请求满仓；受手数、成交量或其他约束时，后续继续尝试买入。
        </p>
      </section>
      <section className="panel">
        <h2>seed {seed} 的策略与基准</h2>
        <MetricsTable experiment={experiment} seed={seed} />
        <p className="hint">
          验证检查点：
          {experiment.runs.find((run) => run.seed === seed)
            ?.checkpoint_selection === "validation_mean_reward"
            ? "验证集平均奖励"
            : "选择口径未知"}
          。测试结果不用于选择 seed。
        </p>
      </section>
      <section className="panel">
        <h2>跨 seed 表现</h2>
        <AggregateTable experiment={experiment} />
        <details>
          <summary>查看所有 seed 的完整指标</summary>
          <MetricsTable experiment={experiment} />
        </details>
      </section>
      <div className="filters">
        <label>
          成交记录策略
          <select
            value={tradePolicy}
            onChange={(event) => setTradePolicy(event.target.value as Policy)}
          >
            {Object.entries(policyNames).map(([value, label]) => (
              <option key={value} value={value}>
                {label}
              </option>
            ))}
          </select>
        </label>
      </div>
      <Trades
        key={`${seed}-${tradePolicy}`}
        id={experiment.experiment_id}
        seed={seed}
        policy={tradePolicy}
      />
    </>
  );
}
export function ExperimentDetail() {
  const { experimentId = "" } = useParams();
  return (
    <ExperimentDetailContent key={experimentId} experimentId={experimentId} />
  );
}
function ExperimentDetailContent({ experimentId }: { experimentId: string }) {
  const navigate = useNavigate();
  const query = useQuery({
    queryKey: ["experiment", experimentId],
    queryFn: ({ signal }) =>
      request<Experiment>(`/experiments/${experimentId}`, { signal }),
  });
  const job = useJob(query.data?.job_id ?? null);
  const [copyError, setCopyError] = useState<Error | null>(null);
  const [downloadError, setDownloadError] = useState<Error | null>(null);
  const [downloading, setDownloading] = useState<string | null>(null);
  const replay = useSubmission(`stockrl.replay.${experimentId}`, (result) =>
    navigate(`/jobs/${result.job_id}`),
  );
  useEffect(() => {
    if (terminal(job.data?.status)) void query.refetch();
  }, [job.data?.status, query.refetch]);
  const item = query.data;
  function copy() {
    if (!item?.request) return;
    try {
      saveStored(DRAFT_KEY, item.request);
      navigate("/experiments/new");
    } catch {
      setCopyError(new Error("无法保存配置草稿，请检查浏览器会话存储。"));
    }
  }
  return (
    <>
      <PageTitle
        eyebrow="RESULT / 实验档案"
        title={item ? experimentName(item) : "实验详情"}
      >
        <Link to="/experiments">返回实验列表</Link>
      </PageTitle>
      <p className="mono break">{experimentId}</p>
      <ErrorBox error={query.error} retry={() => void query.refetch()} />
      {query.isPending && <Loading />}
      {item && (
        <>
          <Provenance experiment={item} />
          {item.kind === "replay" && (
            <p className="notice warning">
              原样重放是可复现性检查，不是新增样本外证据。
              {item.source_experiment_id && (
                <Link to={`/experiments/${item.source_experiment_id}`}>
                  查看来源实验
                </Link>
              )}
            </p>
          )}
          <div className="actions">
            <button
              className="secondary"
              disabled={!item.request || !!replay.pending}
              onClick={copy}
            >
              复制配置创建实验
            </button>
            <button
              disabled={!item.replayable || replay.busy || !!replay.pending}
              onClick={() =>
                replay.start(`/experiments/${experimentId}/replays`, {})
              }
            >
              创建原样重放
            </button>
            {item.job_id && <Link to={`/jobs/${item.job_id}`}>查看任务</Link>}
          </div>
          <ErrorBox error={copyError} />
          {!replay.pending && <ErrorBox error={replay.error} />}
          {!item.replayable && (
            <p className="hint">
              重放不可用：
              {item.replay_block_reason ?? "产物尚未完整或来源未验证。"}
            </p>
          )}
          {replay.pending && (
            <section className="notice warning">
              <p>重放提交等待确认。重试将使用原提交编号，不创建第二个请求。</p>
              <ErrorBox error={replay.error} />
              <div className="actions">
                <button disabled={replay.busy} onClick={replay.retry}>
                  确认 / 重试原重放
                </button>
                <ConfirmAbandon onConfirm={replay.abandon} />
              </div>
            </section>
          )}
          {job.data && !terminal(job.data.status) && (
            <JobProgress job={job.data} />
          )}
          <ErrorBox error={job.error} retry={() => void job.refetch()} />
          <section className="panel">
            <h2>配置与数据指纹</h2>
            {item.dataset ? (
              <DatasetFacts dataset={item.dataset} />
            ) : (
              <p className="hint">
                历史产物未登记应用数据集，计价单位与快照来源可能未知。
              </p>
            )}
            <dl className="facts">
              <div>
                <dt>算法</dt>
                <dd>{algorithm(item)}</dd>
              </div>
              <div>
                <dt>计划训练步数 / seed</dt>
                <dd>
                  {number(
                    item.request?.timesteps ?? item.legacy_config?.timesteps,
                    0,
                  )}
                </dd>
              </div>
              <div>
                <dt>产物 / 核心 / 指标版本</dt>
                <dd>
                  {item.versions.artifact_schema_version} /{" "}
                  {item.versions.core_semantics_version} /{" "}
                  {item.versions.metrics_version}
                </dd>
              </div>
              <div>
                <dt>创建时间</dt>
                <dd>{timestamp(item.created_at)}</dd>
              </div>
              <div className="wide">
                <dt>筛选后数据指纹</dt>
                <dd className="mono break">
                  {item.filtered_data_sha256 ?? "未知"}
                </dd>
              </div>
              <div className="wide">
                <dt>请求指纹</dt>
                <dd className="mono break">
                  {item.request_sha256 ?? "历史产物没有应用请求指纹"}
                </dd>
              </div>
            </dl>
            <div className="split-band">
              {Object.entries(item.splits).map(([name, split]) => (
                <div key={name}>
                  <strong>
                    {{ train: "训练", validation: "验证", test: "测试" }[
                      name
                    ] ?? name}
                  </strong>
                  <span>观察 {split.observation_start_date}</span>
                  <span>
                    收益 {split.first_reward_date} — {split.last_reward_date}
                  </span>
                </div>
              ))}
            </div>
            {item.request?.research_question && (
              <p className="research-question">
                {item.request.research_question}
              </p>
            )}
            <details>
              <summary>
                查看完整提交配置{item.legacy_config && "（历史只读映射）"}
              </summary>
              <pre>
                {JSON.stringify(item.request ?? item.legacy_config, null, 2)}
              </pre>
            </details>
          </section>
          {item.integrity === "complete" ? (
            <ResultViews key={item.experiment_id} experiment={item} />
          ) : (
            <div className="notice warning">
              当前产物尚不满足完整结果展示条件。请查看任务状态或维护诊断。
            </div>
          )}
          <section className="panel">
            <h2>产物下载</h2>
            <ErrorBox error={downloadError} />
            <p className="hint">
              下载的是保存的完整文件；文件名保留 seed
              与策略关系。合成数据、技术验证与重放标记随 manifest / request
              保存。
            </p>
            {!item.artifacts.length && <Empty>暂时没有已登记产物。</Empty>}
            <ul className="artifacts">
              {item.artifacts.map((artifact) => {
                const url = safeDownload(artifact.download_url);
                return (
                  <li key={artifact.artifact_id}>
                    {url ? (
                      <button
                        className="download-link"
                        disabled={downloading === artifact.artifact_id}
                        onClick={async () => {
                          setDownloadError(null);
                          setDownloading(artifact.artifact_id);
                          try {
                            await downloadArtifact(url, artifact.filename);
                          } catch (error) {
                            setDownloadError(error as Error);
                          } finally {
                            setDownloading(null);
                          }
                        }}
                      >
                        {artifact.filename}
                        {downloading === artifact.artifact_id && " · 下载中…"}
                      </button>
                    ) : (
                      <span>{artifact.filename}（下载地址无效）</span>
                    )}
                    <span className="numeric">
                      {number(artifact.size / 1024, 1)} KiB
                    </span>
                    <details>
                      <summary>SHA256</summary>
                      <span className="mono break">{artifact.sha256}</span>
                    </details>
                  </li>
                );
              })}
            </ul>
          </section>
        </>
      )}
    </>
  );
}

export function Compare() {
  const [params, setParams] = useSearchParams();
  const ids = params.getAll("ids");
  const [input, setInput] = useState(ids.join("\n"));
  const query = useQuery({
    queryKey: ["comparison", ids],
    queryFn: ({ signal }) =>
      request<import("../api/contracts").ComparisonResponse>(
        `/comparison?${ids.map((id) => `ids=${encodeURIComponent(id)}`).join("&")}`,
        { signal },
      ),
    enabled: ids.length > 0 && ids.length <= 4,
  });
  return (
    <>
      <PageTitle eyebrow="COMPARE / 条件比较" title="比较实验">
        <Link to="/experiments">从实验列表选择</Link>
      </PageTitle>
      <section className="panel">
        <form
          onSubmit={(event) => {
            event.preventDefault();
            const values = [
              ...new Set(input.split(/[,，\s]+/).filter(Boolean)),
            ];
            setParams(values.map((id): [string, string] => ["ids", id]));
          }}
        >
          <label>
            实验编号（最多 4 个，用逗号或换行分隔）
            <textarea
              value={input}
              onChange={(event) => setInput(event.target.value)}
              rows={2}
            />
          </label>
          <button>读取比较</button>
        </form>
      </section>
      {ids.length > 4 && <p className="notice error">最多选择 4 个实验。</p>}
      {!ids.length && (
        <Empty>从实验列表选择完整的训练实验，或输入实验编号。</Empty>
      )}
      {ids.length > 0 && query.isPending && <Loading />}
      <ErrorBox error={query.error} retry={() => void query.refetch()} />
      {query.data && (
        <>
          <p
            className={`notice ${query.data.same_conditions ? "good" : "warning"}`}
          >
            {query.data.same_conditions
              ? "同条件比较：数据、测试日期、交易配置、单位与语义版本一致。"
              : "条件不同或未确认，以下结果并排展示，不生成合并排名。"}
          </p>
          {query.data.differences.length > 0 && (
            <section className="panel">
              <h2>条件差异</h2>
              {query.data.differences.map((difference, index) => (
                <details key={index}>
                  <summary>{differenceLabel(String(difference.field))}</summary>
                  <div className="comparison-values">
                    {Array.isArray(difference.values) &&
                      difference.values.map((value, valueIndex) => (
                        <div key={valueIndex}>
                          <strong>
                            {query.data.experiments[
                              valueIndex
                            ]?.experiment_id.slice(0, 8)}
                          </strong>
                          <pre>{JSON.stringify(value, null, 2)}</pre>
                        </div>
                      ))}
                  </div>
                </details>
              ))}
            </section>
          )}
          <div className="comparison-grid">
            {query.data.experiments.map((item) => (
              <section className="panel" key={item.experiment_id}>
                <ExperimentLink item={item} />
                <Provenance experiment={item} />
                <dl className="facts">
                  <div>
                    <dt>算法</dt>
                    <dd>{algorithm(item)}</dd>
                  </div>
                  <div>
                    <dt>计划步数 / seed</dt>
                    <dd>
                      {number(
                        item.request?.timesteps ??
                          item.legacy_config?.timesteps,
                        0,
                      )}
                    </dd>
                  </div>
                  <div className="wide">
                    <dt>所有 seed</dt>
                    <dd>
                      {(
                        item.request?.seeds ??
                        item.legacy_config?.seeds ??
                        []
                      ).join(", ")}
                    </dd>
                  </div>
                </dl>
                <p className="hint">
                  计价单位：
                  {item.dataset && item.dataset.quote_unit !== "unknown"
                    ? item.dataset.quote_unit
                    : "未声明计价单位"}
                </p>
                <AggregateTable experiment={item} />
                <details>
                  <summary>所有 seed 指标</summary>
                  <MetricsTable experiment={item} />
                </details>
                <details>
                  <summary>完整配置</summary>
                  <pre>
                    {JSON.stringify(
                      item.request ?? item.legacy_config,
                      null,
                      2,
                    )}
                  </pre>
                </details>
              </section>
            ))}
          </div>
          {query.data.notes.map((note) => (
            <p key={note} className="hint">
              {note}
            </p>
          ))}
        </>
      )}
    </>
  );
}
function differenceLabel(key: string) {
  return (
    (
      {
        filtered_data_sha256: "筛选后数据指纹",
        test_interval: "测试收益日期",
        trading_config: "交易与奖励配置",
        quote_unit: "计价单位",
        core_semantics_version: "核心语义版本",
        metrics_version: "指标版本",
      } as Record<string, string>
    )[key] ?? key
  );
}
