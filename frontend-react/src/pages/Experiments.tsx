import { FormEvent, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { api } from "../api/client";
import { DataTable } from "../components/DataTable";
import { SectionHeader } from "../components/SectionHeader";
import { TerminalPanel } from "../components/TerminalPanel";
import { useT } from "../i18n";

export function Experiments() {
  const t = useT();
  const [compareText, setCompareText] = useState("");
  const experiments = useQuery({ queryKey: ["experiments"], queryFn: api.experiments });
  const compare = useMutation({
    mutationFn: () =>
      api.compareExperiments(
        compareText
          .split(/[,\s]+/)
          .map((item) => item.trim())
          .filter(Boolean)
      )
  });

  function submitCompare(event: FormEvent) {
    event.preventDefault();
    compare.mutate();
  }

  return (
    <>
      <SectionHeader title={t("experiments")} eyebrow={t("experimentManager")} />
      <TerminalPanel title={t("compareExperiments")}>
        <form className="form-stack" onSubmit={submitCompare}>
          <label>
            {t("experimentIds")}
            <textarea value={compareText} onChange={(event) => setCompareText(event.target.value)} rows={3} />
          </label>
          <button className="primary-button" disabled={compare.isPending}>
            {compare.isPending ? t("compare") : t("compare")}
          </button>
        </form>
        <DataTable rows={((compare.data ?? []) as Record<string, unknown>[]).map(flattenExperiment)} />
        {compare.isError ? <div className="inline-error">{(compare.error as Error).message}</div> : null}
      </TerminalPanel>

      <TerminalPanel title={t("experimentList")}>
        <div className="data-table">
          <table>
            <thead>
              <tr>
                <th>ID</th>
                <th>{t("name")}</th>
                <th>{t("status")}</th>
                <th>{t("tags")}</th>
                <th>{t("created")}</th>
              </tr>
            </thead>
            <tbody>
              {(experiments.data?.experiments ?? []).map((item) => (
                <tr key={item.exp_id}>
                  <td>{item.exp_id}</td>
                  <td>{item.name}</td>
                  <td>{item.status ?? "-"}</td>
                  <td>{item.tags.join(", ")}</td>
                  <td>{item.created_at ?? "-"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </TerminalPanel>
    </>
  );
}

function flattenExperiment(value: Record<string, unknown>): Record<string, unknown> {
  const metrics = value.final_metrics && typeof value.final_metrics === "object" ? (value.final_metrics as Record<string, unknown>) : {};
  return {
    exp_id: value.exp_id,
    name: value.name,
    status: value.status,
    tags: Array.isArray(value.tags) ? value.tags.join(", ") : value.tags,
    created_at: value.created_at,
    sharpe: metrics.signal_risk_sharpe_ratio ?? metrics.signal_only_sharpe_ratio,
    return: metrics.signal_risk_annualized_return ?? metrics.signal_only_annualized_return,
    drawdown: metrics.signal_risk_max_drawdown ?? metrics.signal_only_max_drawdown
  };
}
