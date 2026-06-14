import { useT } from "../i18n";

interface RunSelectorProps {
  runs: Record<string, unknown>[];
  value: string;
  onChange: (value: string) => void;
}

export function RunSelector({ runs, value, onChange }: RunSelectorProps) {
  const t = useT();
  return (
    <select value={value} onChange={(event) => onChange(event.target.value)}>
      <option value="">{t("selectRun")}</option>
      {runs.map((run) => {
        const id = String(run.run_id ?? "");
        const ticker = String(run.ticker ?? "");
        return (
          <option key={id} value={id}>
            {ticker ? `${ticker} - ${id}` : id}
          </option>
        );
      })}
    </select>
  );
}
