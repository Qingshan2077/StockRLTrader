import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { api } from "../api/client";
import { DataTable } from "../components/DataTable";
import { EChart } from "../components/EChart";
import { RunSelector } from "../components/RunSelector";
import { SectionHeader } from "../components/SectionHeader";
import { TerminalPanel } from "../components/TerminalPanel";
import { useT } from "../i18n";

export function SignalEvaluation() {
  const t = useT();
  const [runId, setRunId] = useState("");
  const runs = useQuery({ queryKey: ["signal-runs"], queryFn: api.signalRuns });
  const evaluate = useMutation({ mutationFn: () => api.evaluateAlpha(runId, 21) });
  const report = evaluate.data?.report ?? {};
  const summaryRows = toSummaryRows(report);
  const option = {
    backgroundColor: "transparent",
    tooltip: { trigger: "axis" },
    xAxis: { type: "category", data: summaryRows.map((row) => String(row.metric)) },
    yAxis: { type: "value", scale: true },
    series: [{ type: "bar", data: summaryRows.map((row) => Number(row.value ?? 0)), itemStyle: { color: "#55c7e8" } }]
  };

  return (
    <>
      <SectionHeader title={t("signals")} eyebrow={t("alphaQuality")} />
      <TerminalPanel
        title={t("alphaReport")}
        actions={
          <button className="primary-button" onClick={() => evaluate.mutate()} disabled={!runId || evaluate.isPending}>
            {evaluate.isPending ? t("evaluate") : t("evaluate")}
          </button>
        }
      >
        <div className="form-stack">
          <label>
            {t("signalRun")}
            <RunSelector runs={runs.data?.runs ?? []} value={runId} onChange={setRunId} />
          </label>
        </div>
        {summaryRows.length > 0 ? <EChart option={option} height={280} /> : <div className="empty-state">{t("noData")}</div>}
        <DataTable rows={summaryRows} />
        {evaluate.isError ? <div className="inline-error">{(evaluate.error as Error).message}</div> : null}
      </TerminalPanel>
    </>
  );
}

function toSummaryRows(report: Record<string, unknown>): Record<string, unknown>[] {
  const rows: Record<string, unknown>[] = [];
  for (const [section, value] of Object.entries(report)) {
    if (!value || typeof value !== "object" || Array.isArray(value)) {
      continue;
    }
    for (const [metric, metricValue] of Object.entries(value as Record<string, unknown>)) {
      if (typeof metricValue === "number") {
        rows.push({ section, metric, value: Number(metricValue.toFixed(6)) });
      }
    }
  }
  return rows.slice(0, 20);
}
