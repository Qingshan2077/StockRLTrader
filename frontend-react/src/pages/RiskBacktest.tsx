import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { api } from "../api/client";
import { DataTable } from "../components/DataTable";
import { EChart } from "../components/EChart";
import { RunSelector } from "../components/RunSelector";
import { SectionHeader } from "../components/SectionHeader";
import { TerminalPanel } from "../components/TerminalPanel";
import { useT } from "../i18n";
import { useWorkbenchStore } from "../stores/workbenchStore";

export function RiskBacktest() {
  const t = useT();
  const { selectedTicker } = useWorkbenchStore();
  const [runId, setRunId] = useState("");
  const [mode, setMode] = useState("signal_risk");
  const signalRuns = useQuery({ queryKey: ["signal-runs"], queryFn: api.signalRuns });
  const backtestRuns = useQuery({ queryKey: ["backtest-runs"], queryFn: api.backtestRuns, refetchInterval: 3000 });
  const backtest = useMutation({ mutationFn: () => api.runBacktest(selectedTicker, mode, runId || undefined) });
  const result = backtest.data?.result?.result as Record<string, unknown> | undefined;
  const nav = Array.isArray(result?.nav) ? normalizeNav(result.nav as unknown[]) : [];
  const navOption = {
    backgroundColor: "transparent",
    tooltip: { trigger: "axis" },
    xAxis: { type: "category", data: nav.map((row) => String(row.date ?? row.Date ?? "")) },
    yAxis: { type: "value", scale: true },
    series: [{ type: "line", data: nav.map((row) => Number(row.nav ?? row.value ?? 0)), showSymbol: false, lineStyle: { color: "#36c48f" } }]
  };

  return (
    <>
      <SectionHeader title={t("riskBacktest")} eyebrow={t("riskLayer")} />
      <TerminalPanel
        title={t("runBacktest")}
        actions={
          <button className="primary-button" onClick={() => backtest.mutate()} disabled={backtest.isPending}>
            {backtest.isPending ? t("run") : t("runBacktest")}
          </button>
        }
      >
        <div className="form-grid">
          <label>
            {t("signalRun")}
            <RunSelector runs={signalRuns.data?.runs ?? []} value={runId} onChange={setRunId} />
          </label>
          <label>
            {t("mode")}
            <select value={mode} onChange={(event) => setMode(event.target.value)}>
              <option value="signal_only">signal_only</option>
              <option value="signal_risk">signal_risk</option>
              <option value="full">full</option>
            </select>
          </label>
        </div>
        {nav.length > 0 ? <EChart option={navOption} height={300} /> : <div className="empty-state">{t("noData")}</div>}
        <DataTable rows={result?.metrics ? [result.metrics as Record<string, unknown>] : []} />
        {backtest.isError ? <div className="inline-error">{(backtest.error as Error).message}</div> : null}
      </TerminalPanel>

      <TerminalPanel title={t("backtestRuns")}>
        <DataTable rows={((backtestRuns.data?.runs ?? []) as Record<string, unknown>[]).map(flattenBacktestRun)} />
      </TerminalPanel>
    </>
  );
}

function normalizeNav(rows: unknown[]): Record<string, unknown>[] {
  return rows.map((row, index) => {
    if (typeof row === "number") {
      return { date: index, nav: row };
    }
    if (row && typeof row === "object") {
      return row as Record<string, unknown>;
    }
    return { date: index, nav: 0 };
  });
}

function flattenBacktestRun(value: Record<string, unknown>): Record<string, unknown> {
  const metrics = value.metrics && typeof value.metrics === "object" ? (value.metrics as Record<string, unknown>) : {};
  return {
    run_id: value.run_id,
    ticker: value.ticker,
    mode: value.mode,
    created_at: value.created_at,
    annualized_return: metrics.annualized_return,
    sharpe_ratio: metrics.sharpe_ratio,
    max_drawdown: metrics.max_drawdown,
    turnover_rate: metrics.turnover_rate
  };
}
