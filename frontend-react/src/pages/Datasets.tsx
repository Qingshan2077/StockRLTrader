import { useState, type FormEvent } from "react";
import { Link } from "react-router-dom";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { object, request, search, useCapabilities } from "../api/client";
import type {
  Dataset,
  DatasetMetadata,
  DatasetPreview,
  LocalSource,
  Page,
} from "../api/contracts";
import {
  Badge,
  CursorButtons,
  DatasetFacts,
  Empty,
  ErrorBox,
  Loading,
  PageTitle,
  useCursor,
} from "../components";
import { number } from "../display";

export default function Datasets() {
  const client = useQueryClient();
  const caps = useCapabilities();
  const cursor = useCursor();
  const list = useQuery({
    queryKey: ["datasets", cursor.cursor],
    queryFn: ({ signal }) =>
      request<Page<Dataset>>(`/datasets${search({ cursor: cursor.cursor })}`, {
        signal,
      }),
  });
  const sources = useQuery({
    queryKey: ["local-sources"],
    queryFn: ({ signal }) =>
      request<LocalSource[]>("/local-sources", { signal }),
  });
  const [selected, setSelected] = useState<string | null>(null);
  const [mode, setMode] = useState("csv");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const preview = useQuery({
    queryKey: ["dataset-preview", selected],
    queryFn: ({ signal }) =>
      request<DatasetPreview>(`/datasets/${selected}/preview`, { signal }),
    enabled: !!selected,
  });
  const demoDefaults = object(caps.data?.limits.demo_defaults);
  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (busy) return;
    const form = new FormData(event.currentTarget);
    const metadata: DatasetMetadata = {
      display_name: String(form.get("display_name") || "") || null,
      adjustment: String(form.get("adjustment") || "unknown"),
      quote_unit: String(form.get("quote_unit") || "unknown"),
      source_description: String(form.get("source_description") || ""),
    };
    setBusy(true);
    setError(null);
    try {
      let result: Dataset;
      if (mode === "csv") {
        const file = form.get("file");
        if (!(file instanceof File) || !file.size)
          throw new Error("请选择包含 OHLCV 的 CSV 文件。");
        const limit = caps.data?.limits.upload_bytes;
        if (typeof limit === "number" && file.size > limit)
          throw new Error(`文件超过 ${number(limit / 1024 / 1024)} MiB 限制。`);
        const upload = new FormData();
        upload.set("file", file);
        Object.entries(metadata).forEach(([key, value]) => {
          if (value !== null) upload.set(key, value);
        });
        result = await request<Dataset>("/datasets/csv", {
          method: "POST",
          body: upload,
        });
      } else if (mode === "demo")
        result = await request<Dataset>("/datasets/demo", {
          method: "POST",
          body: JSON.stringify({
            ...metadata,
            rows: Number(form.get("rows")),
            seed: Number(form.get("seed")),
          }),
        });
      else
        result = await request<Dataset>("/datasets/local", {
          method: "POST",
          body: JSON.stringify({
            ...metadata,
            source_id: String(form.get("source_id")),
          }),
        });
      setSelected(result.dataset_id);
      cursor.reset();
      await client.invalidateQueries({ queryKey: ["datasets"] });
    } catch (reason) {
      setError(reason instanceof Error ? reason : new Error("数据导入失败。"));
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <PageTitle eyebrow="DATA / 数据快照" title="数据集">
        <p className="hint">导入后保存独立快照，实验使用明确的数据指纹。</p>
        <Link to="/market-datasets">登记研究所需的市场数据与规则</Link>
      </PageTitle>
      <div className="two-column">
        <section className="panel">
          <h2>导入数据</h2>
          <div className="tabs" role="group" aria-label="数据来源">
            {[
              ["csv", "上传 CSV"],
              ["local", "本地原始数据"],
              ["demo", "合成演示"],
            ].map(([value, label]) => (
              <button
                key={value}
                className={mode === value ? "selected" : "secondary"}
                aria-pressed={mode === value}
                disabled={busy}
                onClick={() => setMode(value)}
              >
                {label}
              </button>
            ))}
          </div>
          <form onSubmit={(event) => void submit(event)}>
            <fieldset disabled={busy}>
              <div className="form-grid">
                {mode === "csv" && (
                  <label className="wide">
                    CSV 文件
                    <input
                      name="file"
                      type="file"
                      accept=".csv,text/csv"
                      required
                    />
                    <span className="hint">
                      必须包含
                      Date、Open、High、Low、Close、Volume；日期递增且不重复。
                    </span>
                  </label>
                )}
                {mode === "local" && (
                  <label className="wide">
                    本地原始数据
                    <select name="source_id" required defaultValue="">
                      <option value="" disabled>
                        请选择
                      </option>
                      {sources.data?.map((source) => (
                        <option key={source.source_id} value={source.source_id}>
                          {source.display_name}
                        </option>
                      ))}
                    </select>
                    {sources.isPending && <span>正在查找…</span>}
                    {sources.data?.length === 0 && (
                      <span className="hint">
                        尚无可导入数据，请将原始 CSV 放入配置的本地来源目录。
                      </span>
                    )}
                  </label>
                )}
                {mode === "demo" && (
                  <>
                    <p className="notice warning wide">
                      合成演示仅用于验证流程，不能作为真实市场证据。
                    </p>
                    <label>
                      数据行数
                      <input
                        key={`rows-${demoDefaults.rows}`}
                        name="rows"
                        type="number"
                        min={2}
                        max={
                          typeof caps.data?.limits.dataset_rows === "number"
                            ? caps.data.limits.dataset_rows
                            : undefined
                        }
                        defaultValue={
                          typeof demoDefaults.rows === "number"
                            ? demoDefaults.rows
                            : ""
                        }
                        required
                      />
                    </label>
                    <label>
                      数据生成 seed
                      <input
                        key={`seed-${demoDefaults.seed}`}
                        name="seed"
                        type="number"
                        min={0}
                        max={4294967295}
                        defaultValue={
                          typeof demoDefaults.seed === "number"
                            ? demoDefaults.seed
                            : ""
                        }
                        required
                      />
                    </label>
                  </>
                )}
                <label className="wide">
                  显示名称
                  <input
                    name="display_name"
                    maxLength={200}
                    placeholder="可选，例如 沪深300 日线"
                  />
                </label>
                <label>
                  复权方式
                  <select name="adjustment" defaultValue="unknown">
                    <option value="unknown">未声明</option>
                    <option value="unadjusted">不复权</option>
                    <option value="forward">前复权</option>
                    <option value="backward">后复权</option>
                    <option value="other">其他（请在来源说明填写）</option>
                  </select>
                </label>
                <label>
                  计价单位
                  <input
                    name="quote_unit"
                    maxLength={100}
                    placeholder="未声明；可填写 CNY 等"
                  />
                </label>
                <label className="wide">
                  来源说明
                  <textarea
                    name="source_description"
                    rows={3}
                    maxLength={10000}
                    placeholder="来源、采集时间、价格口径及处理方式"
                  />
                </label>
              </div>
              <button
                type="submit"
                disabled={
                  !caps.data || (mode === "local" && !sources.data?.length)
                }
              >
                {busy ? "正在保存快照…" : "导入为快照"}
              </button>
            </fieldset>
          </form>
          <ErrorBox error={error} />
          <ErrorBox error={caps.error} retry={() => void caps.refetch()} />
          {mode === "local" && (
            <ErrorBox
              error={sources.error}
              retry={() => void sources.refetch()}
            />
          )}
        </section>
        <section className="panel">
          <h2>已保存快照</h2>
          {list.isPending && <Loading />}
          <ErrorBox error={list.error} retry={() => void list.refetch()} />
          {list.data && !list.data.items.length && (
            <Empty>还没有数据集。先导入 CSV 或创建合成演示。</Empty>
          )}
          <ul className="dataset-list">
            {list.data?.items.map((dataset) => (
              <li key={dataset.dataset_id}>
                <button
                  className={`dataset-choice ${selected === dataset.dataset_id ? "active" : ""}`}
                  onClick={() => setSelected(dataset.dataset_id)}
                >
                  <strong>{dataset.display_name}</strong>
                  <span>
                    {dataset.first_date} — {dataset.last_date}
                  </span>
                  <span>
                    {number(dataset.rows, 0)} 行{" "}
                    {dataset.is_synthetic && (
                      <Badge tone="warning">合成演示</Badge>
                    )}
                  </span>
                </button>
              </li>
            ))}
          </ul>
          <CursorButtons
            hasMore={list.data?.has_more ?? false}
            next={() => cursor.next(list.data?.next_cursor ?? null)}
            previous={cursor.previous}
            canPrevious={cursor.canPrevious}
            busy={list.isFetching}
          />
        </section>
      </div>
      {selected && (
        <section className="panel">
          <div className="section-heading">
            <h2>快照预览</h2>
            {preview.data?.trainable && (
              <Link
                className="button"
                to={`/experiments/new?dataset=${selected}`}
              >
                使用该数据创建实验
              </Link>
            )}
          </div>
          {preview.isPending && <Loading />}
          <ErrorBox
            error={preview.error}
            retry={() => void preview.refetch()}
          />
          {preview.data && (
            <>
              <DatasetFacts dataset={preview.data.dataset} />
              {preview.data.warnings.map((warning) => (
                <p key={warning} className="notice warning">
                  {warning}
                </p>
              ))}
              {!preview.data.trainable && (
                <p className="notice warning">
                  当前数据长度不足以满足训练、验证与测试的最小切分。
                </p>
              )}
              <div className="table-scroll">
                <table>
                  <caption>
                    预览 {preview.data.sample.length} /{" "}
                    {preview.data.total_rows} 行
                  </caption>
                  <thead>
                    <tr>
                      {["日期", "开盘", "最高", "最低", "收盘", "成交量"].map(
                        (name) => (
                          <th key={name}>{name}</th>
                        ),
                      )}
                    </tr>
                  </thead>
                  <tbody>
                    {preview.data.sample.map((row) => (
                      <tr key={row.Date}>
                        <td>{row.Date}</td>
                        {[
                          row.Open,
                          row.High,
                          row.Low,
                          row.Close,
                          row.Volume,
                        ].map((value, index) => (
                          <td className="numeric" key={index}>
                            {number(value)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </section>
      )}
    </>
  );
}
