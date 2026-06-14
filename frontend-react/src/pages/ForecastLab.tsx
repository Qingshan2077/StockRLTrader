import { useMutation } from "@tanstack/react-query";
import { api } from "../api/client";
import { DataTable } from "../components/DataTable";
import { EChart } from "../components/EChart";
import { MetricTile } from "../components/MetricTile";
import { SectionHeader } from "../components/SectionHeader";
import { TerminalPanel } from "../components/TerminalPanel";
import { useT } from "../i18n";
import { useWorkbenchStore } from "../stores/workbenchStore";

export function ForecastLab() {
  const t = useT();
  const { selectedTicker } = useWorkbenchStore();
  const probability = useMutation({ mutationFn: () => api.probabilityPrediction(selectedTicker) });
  const advanced = useMutation({ mutationFn: () => api.advancedForecast(selectedTicker, 30) });
  const probEntries = Object.entries(probability.data?.probabilities ?? {});
  const probOption = {
    backgroundColor: "transparent",
    tooltip: { trigger: "axis" },
    xAxis: { type: "category", data: probEntries.map(([horizon]) => `${horizon}D`) },
    yAxis: { type: "value", min: 0, max: 100 },
    series: [{ type: "bar", data: probEntries.map(([, value]) => Number((value * 100).toFixed(2))), itemStyle: { color: "#d6b55f" } }]
  };

  return (
    <>
      <SectionHeader title={t("forecastLab")} eyebrow={t("prediction")} />
      <div className="two-column">
        <TerminalPanel
          title={t("probabilityPrediction")}
          actions={
            <button className="primary-button" onClick={() => probability.mutate()} disabled={probability.isPending}>
              {probability.isPending ? t("train") : `${t("run")} ${selectedTicker}`}
            </button>
          }
        >
          {probability.data ? (
            <>
              <div className="feature-summary">
                {probEntries.map(([horizon, value]) => (
                  <MetricTile key={horizon} label={`${horizon}D ${t("upProb")}`} value={`${(value * 100).toFixed(1)}%`} tone="gold" />
                ))}
              </div>
              <EChart option={probOption} height={280} />
            </>
          ) : (
            <div className="empty-state">{t("noData")}</div>
          )}
          {probability.isError ? <div className="inline-error">{(probability.error as Error).message}</div> : null}
        </TerminalPanel>

        <TerminalPanel
          title={t("advancedForecast")}
          actions={
            <button className="primary-button" onClick={() => advanced.mutate()} disabled={advanced.isPending}>
              {advanced.isPending ? t("train") : t("forecast")}
            </button>
          }
        >
          <DataTable rows={extractForecastRows(advanced.data?.forecast)} maxRows={30} />
          {advanced.isError ? <div className="inline-error">{(advanced.error as Error).message}</div> : null}
        </TerminalPanel>
      </div>
    </>
  );
}

function extractForecastRows(value: unknown): Record<string, unknown>[] {
  if (!value || typeof value !== "object") {
    return [];
  }
  const rows = (value as { rows?: unknown }).rows;
  return Array.isArray(rows) ? (rows as Record<string, unknown>[]) : [];
}
