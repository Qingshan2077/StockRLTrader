import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import {
  object,
  request,
  search,
  terminal,
  usePollInterval,
} from "../api/client";
import {
  economicNames,
  qualificationNames,
  reasonText,
  useResearchSubmission,
  type ResearchDetailData,
} from "../api/research";
import type {
  DiagnosticsReport,
  ResearchReport,
  UnitResult,
} from "../api/research.contracts";
import { ErrorBox, JobProgress, Loading, PageTitle } from "../components";
import { statusNames } from "../display";

export function researchStates(
  technical: string,
  qualification: ResearchReport["qualification"],
  economic: ResearchReport["economic_outcome"],
) {
  return [
    (
      {
        completed: "执行完成",
        failed: "执行失败",
        cancelled: "已取消",
        incomplete: "尚未完整执行",
      } as Record<string, string>
    )[technical] ?? technical,
    qualificationNames[qualification],
    economicNames[economic],
  ];
}
export const diagnosticValue = (value: number | null | undefined) =>
  value == null || !Number.isFinite(value)
    ? "未记录 / 无法计算"
    : `${(value * 100).toFixed(1)}%`;
export const canResumeResearch = (state: string) =>
  ["failed", "cancelled", "interrupted"].includes(state);
const metricNumber = (v: unknown) =>
  typeof v === "number" && Number.isFinite(v) ? v.toFixed(4) : "未记录";
const referenceNames: Record<string, string> = {
  rl: "RL 自主仓位",
  cash: "现金",
  buy_hold: "买入并持有",
  buy_and_hold: "买入并持有",
  fixed_25: "固定 25% 仓位",
  fixed_50: "固定 50% 仓位",
  fixed_75: "固定 75% 仓位",
  fixed_100: "固定 100% 仓位",
  matched_fixed: "风险匹配固定仓位",
  weekly_50: "每周恢复半仓",
  trend_20: "20 日均线趋势",
  vol_target_10: "10% 目标波动率",
  sma_20: "20 日均线",
  momentum_20: "20 日动量",
};
const diagnosticLabels: Record<string, string> = {
  raw_action: "原始动作",
  clipped_action: "裁剪后动作",
  requested_weight: "目标仓位",
  training_log: "优化日志",
  feature_stats: "特征漂移",
  boundary_action_dominance: "动作集中在仓位边界",
  near_half_dominance: "目标仓位集中在半仓附近",
  output_clipping: "存在超出动作范围的输出",
};
export function Diagnostics({
  researchId,
  unit,
}: {
  researchId: string;
  unit: UnitResult;
}) {
  const query = useQuery({
    queryKey: ["research-diagnostics", researchId, unit.key],
    queryFn: ({ signal }) =>
      request<DiagnosticsReport>(
        `/researches/${researchId}/diagnostics${search({ ...unit.key })}`,
        { signal },
      ),
    enabled: unit.status === "completed",
    retry: false,
  });
  if (unit.status !== "completed")
    return <p>此单元尚未完整发布，诊断暂不可用。</p>;
  const d = query.data;
  return (
    <section className="panel">
      <h2>行为与训练诊断</h2>
      <ErrorBox error={query.error} retry={() => void query.refetch()} />
      {query.isPending && <Loading />}
      {d && (
        <>
          <dl className="facts">
            <div>
              <dt>边界动作比例</dt>
              <dd>{diagnosticValue(d.boundary_action_ratio)}</dd>
            </div>
            <div>
              <dt>原始动作越界比例</dt>
              <dd>{diagnosticValue(d.clipping_ratio)}</dd>
            </div>
            <div>
              <dt>接近半仓比例</dt>
              <dd>{diagnosticValue(d.near_half_ratio)}</dd>
            </div>
          </dl>
          {d.warnings.map((w) => (
            <p className="notice warning" key={w}>
              {diagnosticLabels[w] ?? w}
            </p>
          ))}
          {!!d.unavailable.length && (
            <p className="notice warning">
              未记录 / 无法计算：
              {d.unavailable.map((v) => diagnosticLabels[v] ?? v).join("、")}
            </p>
          )}
          <h3>特征分布与漂移</h3>
          {!Object.keys(d.feature_stats).length ? (
            <p>缺少完整特征记录，无法计算漂移。</p>
          ) : (
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>特征</th>
                    <th>原始均值</th>
                    <th>标准化均值</th>
                    <th>裁剪后均值</th>
                    <th>超训练区间</th>
                    <th>特征裁剪比例</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(d.feature_stats).map(([name, stats]) => {
                    const s = object(stats);
                    return (
                      <tr key={name}>
                        <td>{name}</td>
                        <td>{metricNumber(s.raw_mean)}</td>
                        <td>{metricNumber(s.standardized_mean)}</td>
                        <td>{metricNumber(s.clipped_mean)}</td>
                        <td>
                          {typeof s.outside_training_quantiles_ratio ===
                          "number"
                            ? diagnosticValue(
                                s.outside_training_quantiles_ratio,
                              )
                            : "未记录"}
                        </td>
                        <td>
                          {typeof s.clipping_ratio === "number"
                            ? diagnosticValue(s.clipping_ratio)
                            : "未记录"}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
          <h3>优化过程</h3>
          {!d.training_log.length ? (
            <p>优化日志未记录，不据此判断训练收敛。</p>
          ) : (
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>实际步数</th>
                    <th>梯度更新数</th>
                    <th>KL</th>
                    <th>裁剪比例</th>
                    <th>价值损失</th>
                    <th>解释方差</th>
                  </tr>
                </thead>
                <tbody>
                  {d.training_log.map((r, i) => (
                    <tr key={i}>
                      {[
                        "actual_steps",
                        "gradient_updates",
                        "approx_kl",
                        "clip_fraction",
                        "value_loss",
                        "explained_variance",
                      ].map((k) => (
                        <td key={k}>{metricNumber(r[k])}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          <details>
            <summary>行情分组与完整诊断记录</summary>
            <pre>
              {JSON.stringify(
                {
                  regimes: d.regimes,
                  summary: d.summary,
                  training_log: d.training_log,
                },
                null,
                2,
              )}
            </pre>
          </details>
          <p className="hint">
            诊断描述已记录的行为，不自动修改参数，也不单独证明过拟合。
          </p>
        </>
      )}
    </section>
  );
}
export default function ResearchDetail() {
  const { researchId = "" } = useParams(),
    interval = usePollInterval();
  const query = useQuery({
    queryKey: ["research", researchId],
    queryFn: ({ signal }) =>
      request<ResearchDetailData>(`/researches/${researchId}`, { signal }),
    retry: false,
    refetchInterval: (q) =>
      terminal(q.state.data?.job_status.status) ? false : interval(q.state),
  });
  const [selected, setSelected] = useState(""),
    [cancelBusy, setCancelBusy] = useState(false),
    [error, setError] = useState<Error | null>(null);
  const resume = useResearchSubmission(
    `stockrl.research.resume.${researchId}`,
    () => {
      void query.refetch();
    },
  );
  const data = query.data;
  const keyOf = (u: UnitResult) => JSON.stringify(u.key);
  const unit = data?.units.find((u) => keyOf(u) === selected) ?? data?.units[0];
  async function cancel() {
    if (!data || cancelBusy) return;
    setCancelBusy(true);
    setError(null);
    try {
      await request(`/jobs/${data.job_id}/cancel`, {
        method: "POST",
        body: "{}",
      });
      await query.refetch();
    } catch (e) {
      setError(e as Error);
    } finally {
      setCancelBusy(false);
    }
  }
  return (
    <>
      <PageTitle eyebrow="RESEARCH / 证据与诊断" title="研究详情">
        <Link to="/researches">返回研究记录</Link>
      </PageTitle>
      <ErrorBox error={query.error} retry={() => void query.refetch()} />
      <ErrorBox error={error} />
      <ErrorBox error={resume.error} />
      {query.isPending && <Loading />}
      {data && (
        <>
          <h2>{data.protocol.hypothesis}</h2>
          <section className="panel">
            <dl className="facts">
              {researchStates(
                data.technical_status,
                data.qualification,
                data.economic_outcome,
              ).map((value, i) => (
                <div key={i}>
                  <dt>{["技术执行", "研究资格", "经济结论"][i]}</dt>
                  <dd>{value}</dd>
                </div>
              ))}
            </dl>
            {data.qualification === "exploratory" && (
              <p className="notice warning">
                探索性研究：已知信息或测试暴露限制了结论资格，执行完成不会改变这一限制。
              </p>
            )}
            {data.reasons.map((r, i) => (
              <p key={i}>{reasonText(r)}</p>
            ))}
            {data.provisional_outcome && (
              <p>
                仅供描述的暂定结果：{economicNames[data.provisional_outcome]}
              </p>
            )}
          </section>
          <JobProgress job={data.job_status} />
          <div className="actions">
            <button
              className="danger"
              disabled={
                terminal(data.job_status.status) ||
                data.job_status.status === "cancelling" ||
                cancelBusy ||
                !!query.error
              }
              onClick={() => void cancel()}
            >
              {data.job_status.status === "cancelling"
                ? "正在取消，等待计算停止…"
                : "取消本次执行"}
            </button>
            <button
              disabled={
                !canResumeResearch(data.job_status.status) ||
                resume.busy ||
                !!resume.pending ||
                !!query.error
              }
              onClick={() =>
                resume.start(`/researches/${researchId}/resume`, {})
              }
            >
              续跑未完成单元
            </button>
            {resume.pending && (
              <button disabled={resume.busy} onClick={resume.retry}>
                重试同一续跑请求
              </button>
            )}
          </div>
          <p className="hint">
            续跑只执行未完成单元；已完成结果保持不变。浏览器断线不会取消研究。
          </p>
          <section className="panel table-scroll">
            <h2>窗口与随机种子进度</h2>
            <table>
              <thead>
                <tr>
                  <th>标的</th>
                  <th>窗口</th>
                  <th>随机种子</th>
                  <th>状态</th>
                  <th>失败原因</th>
                </tr>
              </thead>
              <tbody>
                {data.unit_progress.map((u) => (
                  <tr key={`${u.instrument_id}:${u.fold_id}:${u.seed}`}>
                    <td>{u.instrument_id}</td>
                    <td>{u.fold_id}</td>
                    <td>{u.seed}</td>
                    <td>{statusNames[u.status] ?? u.status}</td>
                    <td>{u.error_json || "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
          <section className="panel">
            <h2>配对证据汇总</h2>
            <p>
              先汇总同一窗口的随机种子，再比较独立窗口。固定半仓与风险匹配仓位是主要参照。
            </p>
            {!Object.keys(data.assets).length && (
              <p>单元或指标尚未完整，暂不能汇总配对证据。</p>
            )}
            {Object.entries(data.assets).map(([asset, value]) => {
              const a = object(value),
                gates = object(a.gates);
              return (
                <div key={asset}>
                  <h3>{asset}</h3>
                  <p>
                    风险可比比例：
                    {diagnosticValue(
                      typeof a.risk_comparable_fraction === "number"
                        ? a.risk_comparable_fraction
                        : null,
                    )}
                    ；胜出窗口比例：
                    {diagnosticValue(
                      typeof a.winning_fold_fraction === "number"
                        ? a.winning_fold_fraction
                        : null,
                    )}
                  </p>
                  <ul>
                    {Object.entries({
                      positive_base: "基础成本下同时优于两个主要参照",
                      winning_folds: "至少 60% 窗口胜出",
                      drawdown: "最大回撤差不超过 2 个百分点",
                      execution_x2: "两倍执行成本下不落后主要参照",
                    }).map(([key, label]) => (
                      <li key={key}>
                        {label}：
                        {gates[key] === true
                          ? "达到"
                          : gates[key] === false
                            ? "未达到"
                            : "无法判定"}
                      </li>
                    ))}
                  </ul>
                  <div className="table-scroll">
                    <table>
                      <thead>
                        <tr>
                          <th>场景</th>
                          <th>参照</th>
                          <th>年化收益差</th>
                          <th>回撤差</th>
                        </tr>
                      </thead>
                      <tbody>
                        {["base", "execution_x2", "execution_x3"].flatMap((s) =>
                          Object.entries(object(a[s])).map(
                            ([reference, pair]) => {
                              const p = object(pair);
                              return (
                                <tr key={`${s}:${reference}`}>
                                  <td>
                                    {
                                      {
                                        base: "基础成本",
                                        execution_x2: "执行成本 ×2",
                                        execution_x3: "执行成本 ×3",
                                      }[s]
                                    }
                                  </td>
                                  <td>
                                    {referenceNames[reference] ?? reference}
                                  </td>
                                  <td>
                                    {diagnosticValue(
                                      typeof p.cagr_difference === "number"
                                        ? p.cagr_difference
                                        : null,
                                    )}
                                  </td>
                                  <td>
                                    {diagnosticValue(
                                      typeof p.drawdown_difference === "number"
                                        ? p.drawdown_difference
                                        : null,
                                    )}
                                  </td>
                                </tr>
                              );
                            },
                          ),
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>
              );
            })}
            <details>
              <summary>逐标的门槛与配对计算记录</summary>
              <pre>{JSON.stringify(data.assets, null, 2)}</pre>
            </details>
          </section>
          {unit && (
            <>
              <label>
                查看研究单元
                <select
                  value={keyOf(unit)}
                  onChange={(e) => setSelected(e.target.value)}
                >
                  {data.units.map((u) => (
                    <option key={keyOf(u)} value={keyOf(u)}>
                      {u.key.instrument_id} / {u.key.fold_id} / seed{" "}
                      {u.key.seed}
                    </option>
                  ))}
                </select>
              </label>
              <section className="panel">
                <h2>验证选模与测试参照</h2>
                <p>
                  实际训练 {unit.actual_steps.toLocaleString()} 步；选定检查点{" "}
                  {unit.selected_checkpoint_id ?? "尚未选定"}。
                </p>
                <p className="hint">
                  仅使用验证段扣费后表现选模；同分选择较早检查点。测试结果不参与选模。
                </p>
                <p>
                  风险匹配固定仓位：{diagnosticValue(unit.matched_weight)}
                  ；校准误差：{metricNumber(unit.calibration_error)}
                </p>
                <div className="table-scroll">
                  <table>
                    <thead>
                      <tr>
                        <th>检查点</th>
                        <th>实际步数</th>
                        <th>梯度更新</th>
                        <th>验证对数收益</th>
                      </tr>
                    </thead>
                    <tbody>
                      {unit.candidates.map((c) => (
                        <tr key={c.checkpoint_id}>
                          <td>
                            {c.checkpoint_id}
                            {c.checkpoint_id === unit.selected_checkpoint_id
                              ? "（选定）"
                              : ""}
                          </td>
                          <td>{c.actual_steps}</td>
                          <td>{c.gradient_updates}</td>
                          <td>{metricNumber(c.validation_log_return)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                {unit.cost_results.map((c) => (
                  <div key={c.scenario_id}>
                    <h3>
                      {
                        {
                          base: "基础成本",
                          execution_x2: "执行成本 ×2",
                          execution_x3: "执行成本 ×3",
                        }[c.scenario_id]
                      }
                    </h3>
                    <p>
                      测试风险可比：
                      {c.risk_comparable === null
                        ? "无法判定"
                        : c.risk_comparable
                          ? "是"
                          : "否"}
                    </p>
                    <div className="table-scroll">
                      <table>
                        <thead>
                          <tr>
                            <th>策略</th>
                            <th>扣费后收益</th>
                            <th>波动率</th>
                            <th>最大回撤</th>
                            <th>平均仓位</th>
                            <th>费用</th>
                            <th>期末应收</th>
                          </tr>
                        </thead>
                        <tbody>
                          {Object.entries(c.metrics).map(([name, m]) => (
                            <tr key={name}>
                              <td>{referenceNames[name] ?? name}</td>
                              <td>{diagnosticValue(m.net_return)}</td>
                              <td>
                                {diagnosticValue(m.annualized_volatility)}
                              </td>
                              <td>{diagnosticValue(m.max_drawdown)}</td>
                              <td>{diagnosticValue(m.average_exposure)}</td>
                              <td>{metricNumber(m.fees)}</td>
                              <td>{metricNumber(m.final_receivables)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                ))}
              </section>
              <Diagnostics
                key={keyOf(unit)}
                researchId={researchId}
                unit={unit}
              />
            </>
          )}
          <details className="panel">
            <summary>协议、数据指纹与历次执行</summary>
            <p className="mono break">{data.protocol_sha256}</p>
            <pre>
              {JSON.stringify(
                {
                  protocol: data.protocol,
                  preview: data.preview,
                  attempts: data.attempts,
                },
                null,
                2,
              )}
            </pre>
          </details>
        </>
      )}
    </>
  );
}
