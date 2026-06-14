import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { api } from "../api/client";
import { DataTable } from "../components/DataTable";
import { RunSelector } from "../components/RunSelector";
import { SectionHeader } from "../components/SectionHeader";
import { TerminalPanel } from "../components/TerminalPanel";
import { useT } from "../i18n";
import { useWorkbenchStore } from "../stores/workbenchStore";

const MODEL_OPTIONS = ["lightgbm", "ridge", "xgboost", "lasso", "mlp"];

export function AlphaLab() {
  const t = useT();
  const { selectedTicker } = useWorkbenchStore();
  const [models, setModels] = useState(["lightgbm", "ridge"]);
  const [runId, setRunId] = useState("");
  const runs = useQuery({ queryKey: ["signal-runs"], queryFn: api.signalRuns, refetchInterval: 3000 });
  const train = useMutation({ mutationFn: () => api.trainSignals(selectedTicker, models) });
  const ensemble = useMutation({ mutationFn: () => api.buildEnsemble([runId], "weighted") });

  return (
    <>
      <SectionHeader title={t("alphaLab")} eyebrow={t("signalModels")} />
      <div className="two-column">
        <TerminalPanel
          title={t("trainModels")}
          actions={
            <button className="primary-button" onClick={() => train.mutate()} disabled={train.isPending || models.length === 0}>
              {train.isPending ? t("train") : `${t("train")} ${selectedTicker}`}
            </button>
          }
        >
          <div className="check-grid">
            {MODEL_OPTIONS.map((name) => (
              <label key={name}>
                <input
                  type="checkbox"
                  checked={models.includes(name)}
                  onChange={(event) => {
                    setModels((current) => (event.target.checked ? [...current, name] : current.filter((item) => item !== name)));
                  }}
                />
                {name}
              </label>
            ))}
          </div>
          <DataTable rows={train.data ? [flattenTrainResult(train.data)] : []} />
          {train.isError ? <div className="inline-error">{(train.error as Error).message}</div> : null}
        </TerminalPanel>

        <TerminalPanel
          title={t("buildEnsemble")}
          actions={
            <button className="primary-button" onClick={() => ensemble.mutate()} disabled={!runId || ensemble.isPending}>
              {ensemble.isPending ? t("buildEnsemble") : t("weightedEnsemble")}
            </button>
          }
        >
          <div className="form-stack">
            <label>
              {t("signalRun")}
              <RunSelector runs={runs.data?.runs ?? []} value={runId} onChange={setRunId} />
            </label>
          </div>
          <DataTable rows={ensemble.data ? [flattenTrainResult(ensemble.data)] : []} />
          {ensemble.isError ? <div className="inline-error">{(ensemble.error as Error).message}</div> : null}
        </TerminalPanel>
      </div>

      <TerminalPanel title={t("signalRuns")}>
        <DataTable rows={((runs.data?.runs ?? []) as Record<string, unknown>[]).map(flattenSignalRun)} />
      </TerminalPanel>
    </>
  );
}

function flattenTrainResult(value: Record<string, unknown>): Record<string, unknown> {
  return {
    run_id: value.run_id ?? "-",
    job_id: value.job_id ?? "-",
    status: value.status ?? "-",
    result: value.result ? "available" : "-"
  };
}

function flattenSignalRun(value: Record<string, unknown>): Record<string, unknown> {
  const metrics = value.metrics && typeof value.metrics === "object" ? (value.metrics as Record<string, unknown>) : {};
  const firstModel = Array.isArray(value.models) ? String(value.models[0] ?? "") : "";
  const firstMetrics = firstModel && metrics[firstModel] && typeof metrics[firstModel] === "object" ? (metrics[firstModel] as Record<string, unknown>) : {};
  return {
    run_id: value.run_id,
    ticker: value.ticker,
    models: Array.isArray(value.models) ? value.models.join(", ") : value.models,
    created_at: value.created_at,
    val_r2: firstMetrics.val_r2,
    val_rmse: firstMetrics.val_rmse
  };
}
