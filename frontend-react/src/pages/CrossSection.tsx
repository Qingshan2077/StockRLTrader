import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../api/client";
import { DataTable } from "../components/DataTable";
import { EChart } from "../components/EChart";
import { MetricTile } from "../components/MetricTile";
import { SectionHeader } from "../components/SectionHeader";
import { TerminalPanel } from "../components/TerminalPanel";
import { useT } from "../i18n";

export function CrossSection() {
  const t = useT();
  const queryClient = useQueryClient();
  const universe = useQuery({ queryKey: ["cross-section-universe"], queryFn: () => api.crossSectionUniverse(300) });
  const localRank = useQuery({ queryKey: ["cross-section-local-rank"], queryFn: () => api.crossSectionLocalRank(100) });
  const factorReport = useQuery({ queryKey: ["cross-section-factor-report"], queryFn: () => api.crossSectionFactorReport(100) });
  const trainModel = useMutation({
    mutationFn: () => api.trainCrossSection(100, "lightgbm", 5, false),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
    }
  });
  const backtest = useMutation({ mutationFn: () => api.crossSectionLocalBacktest(100, 10, 10) });
  const tickers = universe.data?.tickers ?? [];
  const nav = Array.isArray(backtest.data?.nav) ? (backtest.data.nav as Record<string, unknown>[]) : [];
  const backtestModelStatus = String(backtest.data?.model_status ?? "");
  const backtestMessage = String(backtest.data?.message ?? "");
  const localRankRows = (localRank.data?.records ?? []) as Record<string, unknown>[];
  const localRankColumns = localRank.data?.columns ?? ["date", "code", "prediction", "rank_pct", "future_return", "label_1d"];
  const navOption = {
    backgroundColor: "transparent",
    tooltip: { trigger: "axis" },
    xAxis: { type: "category", data: nav.map((row) => String(row.date ?? "")) },
    yAxis: { type: "value", scale: true },
    series: [{ type: "line", data: nav.map((row) => Number(row.nav ?? 0)), showSymbol: false, lineStyle: { color: "#d6b55f" } }]
  };
  const rankIcRows = factorReport.data?.rank_ic ?? [];
  const rankIcOption = {
    backgroundColor: "transparent",
    tooltip: { trigger: "axis" },
    xAxis: { type: "category", data: rankIcRows.map((row) => String(row.factor ?? "")), axisLabel: { rotate: 30 } },
    yAxis: { type: "value", scale: true },
    series: [{ type: "bar", data: rankIcRows.map((row) => Number(row.rank_ic_mean ?? 0)), itemStyle: { color: "#55c7e8" } }]
  };

  return (
    <>
      <SectionHeader title={t("crossSection")} eyebrow={t("aShare")} />
      <div className="metric-grid">
        <MetricTile label={t("market")} value={universe.data?.market ?? "a_share"} />
        <MetricTile label={t("universeSize")} value={`${universe.data?.count ?? 0}`} tone="gold" />
        <MetricTile label={t("preview")} value={`${tickers.length}`} />
        <MetricTile label={t("modelStatus")} value={localRank.data?.model_status ?? "unknown"} tone={localRank.data?.model_status === "ready" ? "cyan" : "gold"} />
      </div>
      <TerminalPanel title={t("universePreview")}>
        <div className="ticker-cloud">
          {tickers.map((ticker) => (
            <span key={ticker}>{ticker}</span>
          ))}
        </div>
        {universe.isError ? <div className="inline-error">{(universe.error as Error).message}</div> : null}
      </TerminalPanel>

      <div className="two-column">
        <TerminalPanel
          title={t("localRank")}
          actions={
            <button className="primary-button" onClick={() => trainModel.mutate()} disabled={trainModel.isPending}>
              {trainModel.isPending ? t("training") : t("trainCrossSection")}
            </button>
          }
        >
          {localRank.isLoading ? <div className="empty-state">{t("loading")}</div> : null}
          {!localRank.isLoading && localRank.data?.model_status === "missing" ? <div className="empty-state">{localRank.data.message ?? t("trainModelFirst")}</div> : null}
          {!localRank.isLoading && localRankRows.length > 0 ? (
            <DataTable rows={localRankRows} columns={localRankColumns} maxRows={40} />
          ) : null}
          {!localRank.isLoading && localRank.data?.model_status === "ready" && localRankRows.length === 0 ? <div className="empty-state">{localRank.data.message ?? t("noData")}</div> : null}
          {localRank.isError ? <div className="inline-error">{(localRank.error as Error).message}</div> : null}
          {trainModel.isError ? <div className="inline-error">{(trainModel.error as Error).message}</div> : null}
          {trainModel.data ? <div className="empty-state">{`${t("operationSubmitted")} ${trainModel.data.job_id}`}</div> : null}
        </TerminalPanel>

        <TerminalPanel
          title={t("localLongShort")}
          actions={
            <button className="primary-button" onClick={() => backtest.mutate()} disabled={backtest.isPending}>
              {backtest.isPending ? t("run") : t("run")}
            </button>
          }
        >
          {backtest.isPending ? <div className="empty-state">{t("loading")}</div> : null}
          {!backtest.isPending && !backtest.data ? <div className="empty-state">{t("noBacktestYet")}</div> : null}
          {!backtest.isPending && backtestModelStatus === "missing" ? <div className="empty-state">{backtestMessage || t("trainModelFirst")}</div> : null}
          {!backtest.isPending && nav.length > 0 ? <EChart option={navOption} height={260} /> : null}
          {!backtest.isPending && backtest.data && backtestModelStatus !== "missing" && nav.length === 0 ? <div className="empty-state">{backtestMessage || t("noBacktestRows")}</div> : null}
          <DataTable rows={backtest.data?.metrics ? [backtest.data.metrics as Record<string, unknown>] : []} />
          {backtest.isError ? <div className="inline-error">{(backtest.error as Error).message}</div> : null}
        </TerminalPanel>
      </div>

      <TerminalPanel title={t("fullFactorIc")}>
        {factorReport.isLoading ? <div className="empty-state">{t("loading")}</div> : null}
        {!factorReport.isLoading && rankIcRows.length > 0 ? <EChart option={rankIcOption} height={300} /> : null}
        {!factorReport.isLoading && rankIcRows.length === 0 ? <div className="empty-state">{factorReport.data?.message ?? t("noData")}</div> : null}
        <DataTable rows={rankIcRows as Record<string, unknown>[]} />
        {factorReport.isError ? <div className="inline-error">{(factorReport.error as Error).message}</div> : null}
      </TerminalPanel>
    </>
  );
}
