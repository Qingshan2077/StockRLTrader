import { FormEvent, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../api/client";
import { SectionHeader } from "../components/SectionHeader";
import { TerminalPanel } from "../components/TerminalPanel";
import { useT } from "../i18n";

export function DataCenter() {
  const t = useT();
  const queryClient = useQueryClient();
  const [ticker, setTicker] = useState("AAPL");
  const [batchText, setBatchText] = useState("AAPL, MSFT, NVDA");
  const [renameTicker, setRenameTicker] = useState("AAPL");
  const [customName, setCustomName] = useState("");
  const tickers = useQuery({ queryKey: ["tickers"], queryFn: api.tickers });

  const refreshTickers = () => queryClient.invalidateQueries({ queryKey: ["tickers"] });

  const download = useMutation({
    mutationFn: () => api.downloadTicker(ticker, false),
    onSuccess: refreshTickers
  });

  const batchDownload = useMutation({
    mutationFn: () =>
      api.batchDownload(
        batchText
          .split(/[,\s]+/)
          .map((item) => item.trim().toUpperCase())
          .filter(Boolean),
        false
      ),
    onSuccess: refreshTickers
  });

  const rename = useMutation({
    mutationFn: () => api.setCustomName(renameTicker, customName),
    onSuccess: refreshTickers
  });

  const remove = useMutation({
    mutationFn: (symbol: string) => api.deleteTicker(symbol),
    onSuccess: refreshTickers
  });

  function submitDownload(event: FormEvent) {
    event.preventDefault();
    download.mutate();
  }

  function submitBatch(event: FormEvent) {
    event.preventDefault();
    batchDownload.mutate();
  }

  function submitRename(event: FormEvent) {
    event.preventDefault();
    rename.mutate();
  }

  return (
    <>
      <SectionHeader title={t("dataCenter")} eyebrow={t("marketData")} />

      <div className="three-column">
        <TerminalPanel title={t("downloadSymbol")}>
          <form className="form-stack" onSubmit={submitDownload}>
            <label>
              {t("symbol")}
              <input value={ticker} onChange={(event) => setTicker(event.target.value.toUpperCase())} />
            </label>
            <button className="primary-button" disabled={download.isPending}>
              {download.isPending ? t("downloadUpdate") : t("downloadUpdate")}
            </button>
            <MutationMessage mutation={download} />
          </form>
        </TerminalPanel>

        <TerminalPanel title={t("batchDownload")}>
          <form className="form-stack" onSubmit={submitBatch}>
            <label>
              {t("symbol")}
              <textarea value={batchText} onChange={(event) => setBatchText(event.target.value)} rows={3} />
            </label>
            <button className="primary-button" disabled={batchDownload.isPending}>
              {batchDownload.isPending ? t("batchDownload") : t("batchDownload")}
            </button>
            <MutationMessage mutation={batchDownload} />
          </form>
        </TerminalPanel>

        <TerminalPanel title={t("displayName")}>
          <form className="form-stack" onSubmit={submitRename}>
            <label>
              {t("symbol")}
              <input value={renameTicker} onChange={(event) => setRenameTicker(event.target.value.toUpperCase())} />
            </label>
            <label>
              {t("name")}
              <input value={customName} onChange={(event) => setCustomName(event.target.value)} />
            </label>
            <button className="primary-button" disabled={rename.isPending}>
              {t("saveName")}
            </button>
            <MutationMessage mutation={rename} />
          </form>
        </TerminalPanel>
      </div>

      <TerminalPanel title={t("localDataset")}>
        <div className="data-table">
          <table>
            <thead>
              <tr>
                <th>{t("symbol")}</th>
                <th>{t("name")}</th>
                <th>{t("rows")}</th>
                <th>{t("start")}</th>
                <th>{t("end")}</th>
                <th>Raw</th>
                <th>Processed</th>
                <th>{t("actions")}</th>
              </tr>
            </thead>
            <tbody>
              {(tickers.data?.tickers ?? []).map((item) => (
                <tr key={item.ticker}>
                  <td>{item.ticker}</td>
                  <td>{item.custom_name ?? "-"}</td>
                  <td>{item.rows}</td>
                  <td>{item.start?.slice(0, 10) ?? "-"}</td>
                  <td>{item.end?.slice(0, 10) ?? "-"}</td>
                  <td>{item.has_raw ? "yes" : "no"}</td>
                  <td>{item.has_processed ? "yes" : "no"}</td>
                  <td>
                    <button className="danger-button" onClick={() => remove.mutate(item.ticker)} disabled={remove.isPending}>
                      {t("delete")}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </TerminalPanel>
    </>
  );
}

function MutationMessage({ mutation }: { mutation: { isError: boolean; isSuccess: boolean; error: unknown } }) {
  const t = useT();
  if (mutation.isError) {
    return <div className="inline-error">{(mutation.error as Error).message}</div>;
  }
  if (mutation.isSuccess) {
    return <div className="inline-success">{t("operationSubmitted")}</div>;
  }
  return null;
}
