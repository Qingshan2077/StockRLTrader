import { useMutation, useQuery } from "@tanstack/react-query";
import { api } from "../api/client";
import { SectionHeader } from "../components/SectionHeader";
import { TerminalPanel } from "../components/TerminalPanel";
import { useT } from "../i18n";
import { useWorkbenchStore } from "../stores/workbenchStore";

export function Pipeline() {
  const t = useT();
  const { selectedTicker } = useWorkbenchStore();
  const jobs = useQuery({ queryKey: ["jobs"], queryFn: api.jobs, refetchInterval: 3000 });
  const createJob = useMutation({
    mutationFn: () => api.createPipelineJob(selectedTicker)
  });

  return (
    <>
      <SectionHeader title={t("pipeline")} eyebrow={t("endToEnd")} />
      <TerminalPanel
        title={t("runPipeline")}
        actions={
          <button className="primary-button" onClick={() => createJob.mutate()} disabled={createJob.isPending}>
            {createJob.isPending ? t("run") : `${t("run")} ${selectedTicker}`}
          </button>
        }
      >
        <div className="empty-state">Runs signal training, weighted ensemble, backtest comparison, and experiment persistence for the selected symbol.</div>
        {createJob.isError ? <div className="inline-error">{(createJob.error as Error).message}</div> : null}
      </TerminalPanel>

      <TerminalPanel title={t("jobs")}>
        <div className="data-table">
          <table>
            <thead>
              <tr>
                <th>ID</th>
                <th>{t("name")}</th>
                <th>{t("status")}</th>
                <th>{t("progress")}</th>
                <th>{t("message")}</th>
              </tr>
            </thead>
            <tbody>
              {(jobs.data?.jobs ?? []).map((job) => (
                <tr key={job.job_id}>
                  <td>{job.job_id}</td>
                  <td>{job.name}</td>
                  <td>{job.status}</td>
                  <td>{Math.round(job.progress * 100)}%</td>
                  <td>{job.error ?? job.message}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </TerminalPanel>
    </>
  );
}
