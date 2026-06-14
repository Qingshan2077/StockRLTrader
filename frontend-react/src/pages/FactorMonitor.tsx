import { useMutation } from "@tanstack/react-query";
import { api } from "../api/client";
import { DataTable } from "../components/DataTable";
import { EChart } from "../components/EChart";
import { MetricTile } from "../components/MetricTile";
import { SectionHeader } from "../components/SectionHeader";
import { TerminalPanel } from "../components/TerminalPanel";
import { useT } from "../i18n";
import { useWorkbenchStore } from "../stores/workbenchStore";

export function FactorMonitor() {
  const t = useT();
  const { selectedTicker } = useWorkbenchStore();
  const summary = useMutation({ mutationFn: () => api.factorSummary(selectedTicker) });
  const analysis = useMutation({ mutationFn: () => api.factorAnalysis(selectedTicker) });
  const corrRows = Array.isArray(analysis.data?.top_target_correlations)
    ? (analysis.data.top_target_correlations as Record<string, unknown>[]).slice(0, 20)
    : [];
  const corrOption = {
    backgroundColor: "transparent",
    tooltip: { trigger: "axis" },
    xAxis: { type: "category", data: corrRows.map((row) => String(row.feature ?? "")), axisLabel: { rotate: 45 } },
    yAxis: { type: "value", scale: true },
    series: [{ type: "bar", data: corrRows.map((row) => Number(row.target_corr ?? 0)), itemStyle: { color: "#55c7e8" } }]
  };

  return (
    <>
      <SectionHeader title={t("factorMonitor")} eyebrow={t("monitoring")} />
      <TerminalPanel
        title={t("factorSummary")}
        actions={
          <button className="primary-button" onClick={() => summary.mutate()} disabled={summary.isPending}>
            {summary.isPending ? t("scan") : `${t("scan")} ${selectedTicker}`}
          </button>
        }
      >
        {summary.data ? (
          <>
            <div className="feature-summary">
              <MetricTile label={t("features")} value={`${summary.data.feature_count}`} tone="gold" />
              <MetricTile label={t("rows")} value={`${summary.data.row_count}`} />
              <MetricTile label="Groups" value={`${summary.data.groups.length}`} tone="cyan" />
              <MetricTile label={t("cacheHit")} value={`${Math.round(summary.data.cache_hit_rate * 100)}%`} />
            </div>
            <DataTable rows={summary.data.columns.slice(0, 80).map((name, index) => ({ index: index + 1, feature: name }))} />
          </>
        ) : (
          <div className="empty-state">{t("noData")}</div>
        )}
        {summary.isError ? <div className="inline-error">{(summary.error as Error).message}</div> : null}
      </TerminalPanel>

      <TerminalPanel
        title={t("factorAnalysis")}
        actions={
          <button className="primary-button" onClick={() => analysis.mutate()} disabled={analysis.isPending}>
            {analysis.isPending ? t("analyze") : t("analyze")}
          </button>
        }
      >
        {corrRows.length > 0 ? <EChart option={corrOption} height={340} /> : <div className="empty-state">{t("noData")}</div>}
        <DataTable rows={corrRows} />
        {analysis.isError ? <div className="inline-error">{(analysis.error as Error).message}</div> : null}
      </TerminalPanel>
    </>
  );
}
