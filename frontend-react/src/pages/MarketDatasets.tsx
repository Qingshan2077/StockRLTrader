import { useState } from "react";
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { request } from "../api/client";
import {
  type MarketDataset,
  type MarketPreview,
  qualificationNames,
  reasonText,
} from "../api/research";
import { Empty, ErrorBox, Loading, PageTitle } from "../components";

export const bundleFiles = {
  metadata: "metadata.json",
  bars: "bars.csv",
  sessions: "sessions.csv",
  actions: "actions.csv",
  tradability: "tradability.csv",
  profile: "market-profile.json",
};
export function bundleForm(files: Record<string, File>) {
  const form = new FormData();
  for (const key of Object.keys(bundleFiles))
    if (files[key]) form.append(key, files[key]);
  return form;
}
export default function MarketDatasets() {
  const query = useQuery({
    queryKey: ["market-datasets"],
    queryFn: ({ signal }) =>
      request<{ items: MarketDataset[] }>("/market-datasets", { signal }),
  });
  const [files, setFiles] = useState<Record<string, File>>({});
  const [preview, setPreview] = useState<MarketPreview | null>(null),
    [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error | null>(null),
    [registered, setRegistered] = useState(false);
  async function upload(register: boolean) {
    setBusy(true);
    setError(null);
    try {
      const result = await request<MarketPreview>(
        register ? "/market-datasets" : "/market-datasets/preview",
        { method: "POST", body: bundleForm(files) },
      );
      setPreview(result);
      if (register) {
        setRegistered(true);
        await query.refetch();
      }
    } catch (e) {
      setError(e as Error);
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <PageTitle eyebrow="DATA / 研究数据" title="登记市场数据">
        <Link to="/researches/new">创建研究</Link>
      </PageTitle>
      <p>
        导入可审计的原始行情与规则。资料不足可以登记保存，只有“可用于研究”的数据包能够提交研究。
      </p>
      <section className="panel">
        <h2>离线数据包</h2>
        <p className="hint">
          单文件最多 64 MiB，合计最多 100 MiB。旧版数据不会自动转为合格数据。
        </p>
        <fieldset disabled={busy}>
          <div className="form-grid">
            {Object.entries(bundleFiles).map(([field, name]) => (
              <label key={field}>
                {name}
                <input
                  type="file"
                  accept={name.endsWith("csv") ? ".csv" : ".json"}
                  onChange={(event) => {
                    const file = event.target.files?.[0];
                    setFiles((old) => {
                      const next = { ...old };
                      if (file) next[field] = file;
                      else delete next[field];
                      return next;
                    });
                    setPreview(null);
                    setRegistered(false);
                  }}
                />
              </label>
            ))}
          </div>
        </fieldset>
        <div className="actions">
          <button
            disabled={busy || !Object.keys(files).length}
            onClick={() => void upload(false)}
          >
            {busy ? "正在处理…" : "预览数据资格"}
          </button>
          <button
            className="secondary"
            disabled={busy || !preview || registered}
            onClick={() => void upload(true)}
          >
            登记此数据包
          </button>
        </div>
        <ErrorBox error={error} />
        {preview && (
          <div aria-live="polite">
            <h3>{qualificationNames[preview.qualification]}</h3>
            <p>{preview.rows} 行行情</p>
            {[...preview.blocking_reasons, ...preview.issues].map((item, i) => (
              <p key={i} className="notice warning">
                {reasonText(item)}
              </p>
            ))}
            <details>
              <summary>来源与文件指纹</summary>
              <pre>
                {JSON.stringify(
                  {
                    metadata: preview.metadata,
                    file_hashes: preview.file_hashes,
                  },
                  null,
                  2,
                )}
              </pre>
            </details>
          </div>
        )}
        {registered && <p role="status">数据包已登记。</p>}
      </section>
      <ErrorBox error={query.error} retry={() => void query.refetch()} />
      {query.isPending && <Loading />}
      {query.data?.items.length === 0 && (
        <Empty>还没有登记研究数据。请先上传数据包并预览。</Empty>
      )}
      {!!query.data?.items.length && (
        <section className="panel table-scroll">
          <h2>已登记数据</h2>
          <table>
            <thead>
              <tr>
                <th>标的</th>
                <th>覆盖区间</th>
                <th>研究资格</th>
                <th>待补全事项</th>
              </tr>
            </thead>
            <tbody>
              {query.data.items.map((d) => (
                <tr key={d.dataset_id}>
                  <td>
                    {d.metadata
                      ? `${d.metadata.exchange}:${d.metadata.symbol}`
                      : d.dataset_id.slice(0, 8)}
                  </td>
                  <td>
                    {d.metadata?.coverage_start ?? "未知"} —{" "}
                    {d.metadata?.coverage_end ?? "未知"}
                  </td>
                  <td>{qualificationNames[d.qualification]}</td>
                  <td>
                    {d.blocking_reasons.map(reasonText).join("；") || "无"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}
      <Link to="/datasets">查看旧版数据集</Link>
    </>
  );
}
