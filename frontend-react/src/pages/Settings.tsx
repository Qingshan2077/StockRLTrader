import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client";
import { DataTable } from "../components/DataTable";
import { MetricTile } from "../components/MetricTile";
import { SectionHeader } from "../components/SectionHeader";
import { TerminalPanel } from "../components/TerminalPanel";
import { useT } from "../i18n";
import { useLanguageStore } from "../stores/languageStore";
import type { Language } from "../i18n";

export function Settings() {
  const t = useT();
  const language = useLanguageStore((state) => state.language);
  const setLanguage = useLanguageStore((state) => state.setLanguage);
  const summary = useQuery({ queryKey: ["system-summary"], queryFn: api.systemSummary });
  const config = useQuery({ queryKey: ["system-config"], queryFn: api.systemConfig });

  return (
    <>
      <SectionHeader title={t("settings")} eyebrow={t("system")} />
      <div className="metric-grid">
        <MetricTile label={t("system")} value={summary.data?.name ?? "--"} />
        <MetricTile label={t("version")} value={summary.data?.version ?? "--"} tone="gold" />
        <MetricTile label={t("stage")} value={summary.data?.stage ?? "--"} />
        <MetricTile label={t("dataDir")} value={summary.data?.data_dir ?? "--"} tone="cyan" />
      </div>

      <TerminalPanel title={t("language")}>
        <div className="form-stack">
          <label>
            {t("language")}
            <select value={language} onChange={(event) => setLanguage(event.target.value as Language)}>
              <option value="en">{t("english")}</option>
              <option value="zh">{t("chinese")}</option>
            </select>
          </label>
        </div>
      </TerminalPanel>

      <TerminalPanel title={t("runtimeSummary")}>
        <DataTable rows={summary.data ? [summary.data as unknown as Record<string, unknown>] : []} />
      </TerminalPanel>

      <TerminalPanel title={t("mergedConfig")}>
        <DataTable rows={config.data ? flattenConfig(config.data) : []} />
      </TerminalPanel>
    </>
  );
}

function flattenConfig(config: Record<string, unknown>): Record<string, unknown>[] {
  return Object.entries(config).map(([key, value]) => ({
    key,
    type: Array.isArray(value) ? "array" : typeof value,
    value: typeof value === "object" && value !== null ? Object.keys(value as Record<string, unknown>).join(", ") : value
  }));
}
