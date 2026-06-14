import { useMemo } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { api } from "../api/client";
import { CandlestickChart } from "../components/CandlestickChart";
import { MetricTile } from "../components/MetricTile";
import { SectionHeader } from "../components/SectionHeader";
import { TerminalPanel } from "../components/TerminalPanel";
import { useT } from "../i18n";
import { useWorkbenchStore } from "../stores/workbenchStore";

export function Dashboard() {
  const t = useT();
  const { selectedTicker, setSelectedTicker } = useWorkbenchStore();
  const tickers = useQuery({ queryKey: ["tickers"], queryFn: api.tickers });
  const candles = useQuery({
    queryKey: ["candles", selectedTicker],
    queryFn: () => api.candles(selectedTicker, 600),
    enabled: Boolean(selectedTicker)
  });
  const featureSummary = useMutation({
    mutationFn: () => api.buildFeatures(selectedTicker)
  });

  const latest = candles.data?.records.at(-1);
  const previous = candles.data?.records.at(-2);
  const priceChange =
    typeof latest?.Close === "number" && typeof previous?.Close === "number"
      ? ((latest.Close - previous.Close) / previous.Close) * 100
      : null;

  const tickerOptions = useMemo(() => tickers.data?.tickers ?? [], [tickers.data]);

  return (
    <>
      <SectionHeader title={t("dashboard")} eyebrow={t("market")} />
      <div className="toolbar">
        <label>
          {t("symbol")}
          <select value={selectedTicker} onChange={(event) => setSelectedTicker(event.target.value)}>
            {tickerOptions.length === 0 ? <option value={selectedTicker}>{selectedTicker}</option> : null}
            {tickerOptions.map((item) => (
              <option key={item.ticker} value={item.ticker}>
                {item.custom_name ? `${item.custom_name} (${item.ticker})` : item.ticker}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div className="metric-grid">
        <MetricTile label={t("lastPrice")} value={typeof latest?.Close === "number" ? latest.Close.toFixed(2) : "--"} />
        <MetricTile
          label={t("dailyChange")}
          value={priceChange === null ? "--" : `${priceChange.toFixed(2)}%`}
          tone={priceChange === null ? "neutral" : priceChange >= 0 ? "bull" : "bear"}
        />
        <MetricTile label={t("volume")} value={typeof latest?.Volume === "number" ? `${(latest.Volume / 1e6).toFixed(1)}M` : "--"} />
        <MetricTile label={t("source")} value={candles.data?.source ?? "--"} detail={`${candles.data?.row_count ?? 0} ${t("rows")}`} tone="cyan" />
      </div>

      <TerminalPanel title={`${selectedTicker} Candles`}>
        {candles.isError ? <div className="empty-state">{(candles.error as Error).message}</div> : null}
        {candles.data ? <CandlestickChart records={candles.data.records} /> : <div className="empty-state">{t("loadingMarket")}</div>}
      </TerminalPanel>

      <div className="two-column">
        <TerminalPanel
          title={t("featureSummary")}
          actions={
            <button className="ghost-button" onClick={() => featureSummary.mutate()} disabled={featureSummary.isPending}>
              {featureSummary.isPending ? t("buildFeatures") : t("buildFeatures")}
            </button>
          }
        >
          {featureSummary.data ? (
            <div className="feature-summary">
              <MetricTile label={t("features")} value={`${featureSummary.data.feature_count}`} tone="gold" />
              <MetricTile label={t("rows")} value={`${featureSummary.data.row_count}`} />
              <MetricTile label={t("labels")} value={`${featureSummary.data.label_count}`} />
              <MetricTile label={t("cacheHit")} value={`${Math.round(featureSummary.data.cache_hit_rate * 100)}%`} tone="cyan" />
            </div>
          ) : (
            <div className="empty-state">{t("buildFeatureHint")}</div>
          )}
          {featureSummary.isError ? <div className="inline-error">{(featureSummary.error as Error).message}</div> : null}
        </TerminalPanel>

        <TerminalPanel title={t("status")}>
          <div className="status-list">
            <div>
              <span>{t("symbol")}</span>
              <strong>{selectedTicker}</strong>
            </div>
            <div>
              <span>{t("records")}</span>
              <strong>{candles.data?.row_count ?? 0}</strong>
            </div>
            <div>
              <span>{t("localFile")}</span>
              <strong>{candles.data?.source ?? "--"}</strong>
            </div>
          </div>
        </TerminalPanel>
      </div>
    </>
  );
}
