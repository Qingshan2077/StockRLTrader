import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { request } from "../api/client";
import {
  canQueue,
  datasetFields,
  qualificationNames,
  reasonText,
  useResearchSubmission,
  type Draft,
  type MarketDataset,
  type Preview,
} from "../api/research";
import { ErrorBox, Loading, PageTitle } from "../components";

export default function NewResearch() {
  const navigate = useNavigate();
  const datasets = useQuery({
    queryKey: ["market-datasets"],
    queryFn: ({ signal }) =>
      request<{ items: MarketDataset[] }>("/market-datasets", { signal }),
  });
  const [selected, setSelected] = useState<string[]>([]);
  const [form, setForm] = useState({
    protocol_id: "",
    hypothesis: "RL 自主仓位能否在相近风险下，比透明基准提供稳定的扣费后改善？",
    asset_selection_note: "",
    first_test_session: "",
    fold_count: 4,
    seeds: "42,43,44,45,46",
    requested_timesteps: 100000,
    episode_length: 126,
    algorithm: "PPO" as "PPO" | "SAC",
    qualification: "exploratory" as Draft["qualification"],
  });
  const [preview, setPreview] = useState<{
    request: string;
    value: Preview;
  } | null>(null);
  const [busy, setBusy] = useState(false),
    [error, setError] = useState<Error | null>(null);
  const submission = useResearchSubmission(
    "stockrl.research.submit",
    (result) => navigate(`/researches/${result.research_id}`),
  );
  let draft: Draft | null = null,
    invalid = "";
  try {
    const chosen = (datasets.data?.items ?? []).filter((d) =>
      selected.includes(d.dataset_id),
    );
    const fields = datasetFields(chosen);
    const seeds = form.seeds
      .split(/[,，\s]+/)
      .filter(Boolean)
      .map(Number);
    if (
      !seeds.length ||
      seeds.some(
        (seed) => !Number.isInteger(seed) || seed < 0 || seed > 4294967295,
      ) ||
      new Set(seeds).size !== seeds.length
    )
      throw new Error("随机种子需为不重复的 0–4294967295 整数。");
    if (
      !form.protocol_id.trim() ||
      !form.hypothesis.trim() ||
      !form.asset_selection_note.trim() ||
      !form.first_test_session
    )
      throw new Error("请填写协议编号、研究命题、选股理由和首个测试日期。");
    if (
      !Number.isInteger(form.fold_count) ||
      form.fold_count < 1 ||
      !Number.isInteger(form.requested_timesteps) ||
      form.requested_timesteps < 1 ||
      form.requested_timesteps > 1000000 ||
      !Number.isInteger(form.episode_length) ||
      form.episode_length < 1
    )
      throw new Error(
        "窗口数与回合长度需为正整数，每单元预算最多 1,000,000 步。",
      );
    draft = {
      ...fields,
      scope: chosen.length === 1 ? "single_asset" : "asset_set",
      protocol_id: form.protocol_id.trim(),
      hypothesis: form.hypothesis.trim(),
      asset_selection_note: form.asset_selection_note.trim(),
      first_test_session: form.first_test_session,
      fold_count: form.fold_count,
      seeds,
      qualification: form.qualification,
      training_budget: {
        requested_timesteps: form.requested_timesteps,
        episode_length: form.episode_length,
        algorithm: form.algorithm,
      },
    };
  } catch (e) {
    invalid = (e as Error).message;
  }
  function change<K extends keyof typeof form>(
    key: K,
    value: (typeof form)[K],
  ) {
    setForm((old) => ({ ...old, [key]: value }));
    setPreview(null);
    setError(null);
  }
  async function inspect() {
    if (!draft || busy) return;
    const payload = JSON.stringify(draft);
    setBusy(true);
    setPreview(null);
    setError(null);
    try {
      const value = await request<Preview>("/research-previews", {
        method: "POST",
        body: payload,
      });
      setPreview({ request: payload, value });
    } catch (e) {
      setError(e as Error);
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <PageTitle eyebrow="RESEARCH / 预注册" title="创建研究">
        <Link to="/researches">返回研究记录</Link>
      </PageTitle>
      <p>
        先确认研究问题与数据，再核对窗口、预算和测试暴露。预览不会创建任务。
      </p>
      <ErrorBox error={datasets.error} retry={() => void datasets.refetch()} />
      {datasets.isPending && <Loading />}
      <fieldset disabled={busy || submission.busy || !!submission.pending}>
        <section className="panel">
          <h2>数据与研究问题</h2>
          <p>
            <Link to="/market-datasets">登记新的市场数据包</Link>
          </p>
          {(datasets.data?.items ?? []).map((d) => (
            <label className="check" key={d.dataset_id}>
              <input
                type="checkbox"
                disabled={d.qualification !== "ready"}
                checked={selected.includes(d.dataset_id)}
                onChange={(e) => {
                  setSelected((old) =>
                    e.target.checked
                      ? [...old, d.dataset_id]
                      : old.filter((id) => id !== d.dataset_id),
                  );
                  setPreview(null);
                }}
              />
              {d.metadata
                ? `${d.metadata.exchange}:${d.metadata.symbol}`
                : d.dataset_id.slice(0, 8)}{" "}
              · {qualificationNames[d.qualification]}
              {d.qualification !== "ready" && (
                <small>{d.blocking_reasons.map(reasonText).join("；")}</small>
              )}
            </label>
          ))}
          {datasets.data?.items.length === 0 && (
            <p>请先登记至少一个可用于研究的数据包。</p>
          )}
          <div className="form-grid">
            <label>
              协议编号
              <input
                value={form.protocol_id}
                onChange={(e) => change("protocol_id", e.target.value)}
              />
            </label>
            <label>
              首个测试日期
              <input
                type="date"
                value={form.first_test_session}
                onChange={(e) => change("first_test_session", e.target.value)}
              />
            </label>
            <label>
              研究命题
              <textarea
                value={form.hypothesis}
                onChange={(e) => change("hypothesis", e.target.value)}
              />
            </label>
            <label>
              选股理由
              <textarea
                value={form.asset_selection_note}
                onChange={(e) => change("asset_selection_note", e.target.value)}
              />
            </label>
          </div>
        </section>
        <section className="panel">
          <h2>窗口与执行预算</h2>
          <div className="form-grid">
            <label>
              滚动窗口数
              <input
                type="number"
                min="1"
                value={form.fold_count}
                onChange={(e) => change("fold_count", Number(e.target.value))}
              />
            </label>
            <label>
              随机种子
              <input
                value={form.seeds}
                onChange={(e) => change("seeds", e.target.value)}
              />
              <small>逗号分隔。种子数量不增加独立市场窗口数。</small>
            </label>
            <label>
              每单元请求步数
              <input
                type="number"
                min="1"
                max="1000000"
                value={form.requested_timesteps}
                onChange={(e) =>
                  change("requested_timesteps", Number(e.target.value))
                }
              />
            </label>
            <label>
              训练回合长度
              <input
                type="number"
                min="1"
                value={form.episode_length}
                onChange={(e) =>
                  change("episode_length", Number(e.target.value))
                }
              />
            </label>
            <label>
              算法
              <select
                value={form.algorithm}
                onChange={(e) =>
                  change("algorithm", e.target.value as "PPO" | "SAC")
                }
              >
                <option>PPO</option>
                <option>SAC</option>
              </select>
            </label>
            <label>
              研究资格声明
              <select
                value={form.qualification}
                onChange={(e) =>
                  change(
                    "qualification",
                    e.target.value as Draft["qualification"],
                  )
                }
              >
                <option value="exploratory">探索性研究</option>
                <option value="declared_holdout">预先声明的留出研究</option>
              </select>
            </label>
          </div>
          <p className="hint">
            已查看的测试区间不会因更换编号或随机种子而恢复为留出数据。费用场景和验收标准按协议固定。
          </p>
        </section>
      </fieldset>
      {invalid && <p className="notice warning">{invalid}</p>}
      <ErrorBox error={error} />
      <ErrorBox error={submission.error} />
      <button
        disabled={!draft || busy || !!submission.pending}
        onClick={() => void inspect()}
      >
        {busy ? "正在计算预览…" : "预览预算与暴露"}
      </button>
      {preview && (
        <section className="panel" aria-live="polite">
          <h2>提交前确认</h2>
          <p className="notice warning">
            {qualificationNames[preview.value.qualification]}
            。技术执行成功不等于发现投资优势。
          </p>
          <dl className="facts">
            <div>
              <dt>训练单元</dt>
              <dd>{preview.value.budget.unit_count}</dd>
            </div>
            <div>
              <dt>总请求步数</dt>
              <dd>
                {preview.value.budget.requested_total_steps.toLocaleString()}
              </dd>
            </div>
            <div>
              <dt>实际训练步数上限</dt>
              <dd>
                {preview.value.budget.rollout_upper_bound.toLocaleString()}
              </dd>
            </div>
            <div>
              <dt>额外评估次数</dt>
              <dd>{preview.value.budget.extra_evaluations}</dd>
            </div>
            <div>
              <dt>运行时间上限</dt>
              <dd>
                {(preview.value.maximum_runtime_seconds / 3600).toFixed(1)} 小时
              </dd>
            </div>
          </dl>
          {preview.value.exposure_warnings.length ? (
            preview.value.exposure_warnings.map((reason) => (
              <p key={reason} className="notice warning">
                {reasonText(reason)}
              </p>
            ))
          ) : (
            <p>没有发现已登记的测试区间重叠。资格仍以协议与来源记录为准。</p>
          )}
          {preview.value.blockers.map((reason) => (
            <p role="alert" className="notice error" key={reason}>
              {reasonText(reason)}
            </p>
          ))}
          {!!preview.value.omitted_folds.length && (
            <p>未纳入的窗口：{preview.value.omitted_folds.join("；")}</p>
          )}
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>标的 / 窗口</th>
                  <th>训练</th>
                  <th>验证</th>
                  <th>测试</th>
                </tr>
              </thead>
              <tbody>
                {preview.value.fold_plan.map((f) => (
                  <tr key={`${f.instrument_id}:${f.fold_id}`}>
                    <td>
                      {f.instrument_id} / {f.fold_id}
                    </td>
                    {[f.train, f.validation, f.test].map((w, i) => (
                      <td key={i}>
                        {w.start_session} — {w.end_session}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <details>
            <summary>检查点与协议指纹</summary>
            <p className="break mono">{preview.value.protocol_sha256}</p>
            <p>{preview.value.checkpoint_steps.join("、")}</p>
          </details>
        </section>
      )}
      <div className="actions">
        <button
          disabled={
            !canQueue(draft, preview) ||
            submission.busy ||
            !!submission.pending ||
            busy
          }
          onClick={() => {
            if (canQueue(draft, preview))
              submission.start("/researches", draft);
          }}
        >
          确认预算并排队
        </button>
        {submission.pending && (
          <button disabled={submission.busy} onClick={submission.retry}>
            重试同一提交
          </button>
        )}
        {submission.rejected && (
          <button
            className="secondary"
            onClick={() => {
              submission.editRejected();
              setPreview(null);
            }}
          >
            修改被拒绝的请求
          </button>
        )}
      </div>
      {submission.pending && (
        <p className="notice warning">
          提交结果尚未确认。重试将沿用原请求与编号，避免重复排队；刷新页面不会丢失该请求。
        </p>
      )}
    </>
  );
}
