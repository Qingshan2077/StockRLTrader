import { useMutation, useQuery } from "@tanstack/react-query";
import { api } from "../api/client";
import { DataTable } from "../components/DataTable";
import { EChart } from "../components/EChart";
import { MetricTile } from "../components/MetricTile";
import { SectionHeader } from "../components/SectionHeader";
import { TerminalPanel } from "../components/TerminalPanel";
import { useT } from "../i18n";

export function CrossSection() {
  const t = useT();
  const universe = useQuery({ queryKey: ["cross-section-universe"], queryFn: () => api.crossSectionUniverse(300) });
  const localRank = useQuery({ queryKey: ["cross-section-local-rank"], queryFn: () => api.crossSectionLocalRank(100) });
  const factorReport = useQuery({ queryKey: ["cross-section-factor-report"], queryFn: () => api.crossSectionFactorReport(100) });
  const backtest = useMutation({ mutationFn: () => api.crossSectionLocalBacktest(100, 10, 10) });
  const tickers = universe.data?.tickers ?? [];
  const nav = Array.isArray(backtest.data?.nav) ? (backtest.data.nav as Record<string, unknown>[]) : [];
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
        <MetricTile label={t("status")} value={universe.isError ? "error" : t("ready")} tone={universe.isError ? "bear" : "cyan"} />
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
        <TerminalPanel title={t("localRank")}>
          <DataTable
            rows={(localRank.data?.records ?? []) as Record<string, unknown>[]}
            columns={["date", "code", "multi_factor_score", "rank_pct", "momentum_20", "low_vol_20", "liquidity_20", "future_return"]}
            maxRows={40}
          />
          {localRank.isError ? <div className="inline-error">{(localRank.error as Error).message}</div> : null}
        </TerminalPanel>

        <TerminalPanel
          title={t("localLongShort")}
          actions={
            <button className="primary-button" onClick={() => backtest.mutate()} disabled={backtest.isPending}>
              {backtest.isPending ? t("run") : t("run")}
            </button>
          }
        >
          {nav.length > 0 ? <EChart option={navOption} height={260} /> : <div className="empty-state">{t("noData")}</div>}
          <DataTable rows={backtest.data?.metrics ? [backtest.data.metrics as Record<string, unknown>] : []} />
          {backtest.isError ? <div className="inline-error">{(backtest.error as Error).message}</div> : null}
        </TerminalPanel>
      </div>

      <TerminalPanel title={`${t("factor")} ${t("rankIc")}`}>
        {rankIcRows.length > 0 ? <EChart option={rankIcOption} height={300} /> : <div className="empty-state">{t("noData")}</div>}
        <DataTable rows={rankIcRows as Record<string, unknown>[]} />
        {factorReport.isError ? <div className="inline-error">{(factorReport.error as Error).message}</div> : null}
      </TerminalPanel>
    </>
  );
}
