import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { api } from "../api/client";
import { DataTable } from "../components/DataTable";
import { RunSelector } from "../components/RunSelector";
import { SectionHeader } from "../components/SectionHeader";
import { TerminalPanel } from "../components/TerminalPanel";
import { useT } from "../i18n";
import { useWorkbenchStore } from "../stores/workbenchStore";

export function ExecutionLab() {
  const t = useT();
  const { selectedTicker } = useWorkbenchStore();
  const [algorithm, setAlgorithm] = useState("PPO");
  const [runId, setRunId] = useState("");
  const jobs = useQuery({ queryKey: ["jobs"], queryFn: api.jobs, refetchInterval: 3000 });
  const signalRuns = useQuery({ queryKey: ["signal-runs"], queryFn: api.signalRuns });
  const train = useMutation({ mutationFn: () => api.trainRl(selectedTicker, algorithm, 50000, runId || undefined) });

  return (
    <>
      <SectionHeader title={t("execution")} eyebrow={t("rlExecution")} />
      <TerminalPanel
        title={t("trainRlAgent")}
        actions={
          <button className="primary-button" onClick={() => train.mutate()} disabled={train.isPending}>
            {train.isPending ? t("train") : `${t("train")} ${algorithm}`}
          </button>
        }
      >
        <div className="form-stack">
          <label>
            {t("algorithm")}
            <select value={algorithm} onChange={(event) => setAlgorithm(event.target.value)}>
              <option value="PPO">PPO</option>
              <option value="SAC">SAC</option>
            </select>
          </label>
          <label>
            {t("signalRun")}
            <RunSelector runs={signalRuns.data?.runs ?? []} value={runId} onChange={setRunId} />
          </label>
        </div>
        <DataTable rows={train.data ? [flattenRlResponse(train.data as unknown as Record<string, unknown>)] : []} />
        {train.isError ? <div className="inline-error">{(train.error as Error).message}</div> : null}
      </TerminalPanel>

      <TerminalPanel title={t("executionJobs")}>
        <DataTable rows={(jobs.data?.jobs ?? []).filter((job) => job.name.startsWith("rl:")) as unknown as Record<string, unknown>[]} />
      </TerminalPanel>
    </>
  );
}

function flattenRlResponse(value: Record<string, unknown>): Record<string, unknown> {
  return {
    job_id: value.job_id,
    status: value.status,
    run_id: value.run_id ?? "-"
  };
}
