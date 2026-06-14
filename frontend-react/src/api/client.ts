import type {
  AdvancedForecastResponse,
  AlphaEvaluationResponse,
  BacktestRunListResponse,
  BacktestRunResponse,
  BatchDownloadResponse,
  CandleResponse,
  CrossSectionRankResponse,
  CrossSectionFactorReportResponse,
  DeleteTickerResponse,
  DownloadResponse,
  ExperimentListResponse,
  ExperimentCompareResponse,
  FeatureSummaryResponse,
  JobListResponse,
  JobResponse,
  ProbabilityPredictionResponse,
  SignalRunListResponse,
  SignalRunResponse,
  SystemConfig,
  SystemSummary,
  TickerListResponse,
  UniverseResponse
} from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {})
    },
    ...init
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail ?? `Request failed: ${response.status}`);
  }

  return response.json() as Promise<T>;
}

export const api = {
  systemSummary: () => request<SystemSummary>("/api/system/summary"),
  systemConfig: () => request<SystemConfig>("/api/system/config"),
  tickers: () => request<TickerListResponse>("/api/market/tickers"),
  candles: (ticker: string, limit = 500) =>
    request<CandleResponse>(`/api/market/${encodeURIComponent(ticker)}/candles?limit=${limit}`),
  downloadTicker: (ticker: string, forceUpdate = false, startDate?: string) =>
    request<DownloadResponse>(`/api/market/${encodeURIComponent(ticker)}/download`, {
      method: "POST",
      body: JSON.stringify({ force_update: forceUpdate, start_date: startDate })
    }),
  batchDownload: (tickers: string[], forceUpdate = false, startDate?: string) =>
    request<BatchDownloadResponse>("/api/market/batch-download", {
      method: "POST",
      body: JSON.stringify({ tickers, force_update: forceUpdate, start_date: startDate })
    }),
  setCustomName: (ticker: string, customName: string) =>
    request(`/api/market/${encodeURIComponent(ticker)}/custom-name`, {
      method: "POST",
      body: JSON.stringify({ custom_name: customName })
    }),
  deleteTicker: (ticker: string) =>
    request<DeleteTickerResponse>(`/api/market/${encodeURIComponent(ticker)}`, {
      method: "DELETE"
    }),
  buildFeatures: (ticker: string) =>
    request<FeatureSummaryResponse>(`/api/features/${encodeURIComponent(ticker)}/build`, {
      method: "POST",
      body: JSON.stringify({ use_cache: true, horizon: 5, label_type: "regression" })
    }),
  probabilityPrediction: (ticker: string, horizons = [1, 5, 10]) =>
    request<ProbabilityPredictionResponse>("/api/prediction/probability", {
      method: "POST",
      body: JSON.stringify({ ticker, horizons })
    }),
  advancedForecast: (ticker: string, days = 30) =>
    request<AdvancedForecastResponse>("/api/prediction/advanced/forecast", {
      method: "POST",
      body: JSON.stringify({ ticker, days })
    }),
  signalRuns: () => request<SignalRunListResponse>("/api/signals/runs"),
  trainSignals: (ticker: string, models: string[], horizon = 5) =>
    request<SignalRunResponse>("/api/signals/train", {
      method: "POST",
      body: JSON.stringify({ ticker, models, horizon, label_type: "regression", use_cache: true })
    }),
  buildEnsemble: (modelRunIds: string[], method = "weighted") =>
    request<SignalRunResponse>("/api/signals/ensemble", {
      method: "POST",
      body: JSON.stringify({ method, model_run_ids: modelRunIds })
    }),
  evaluateAlpha: (modelRunId: string, window = 21) =>
    request<AlphaEvaluationResponse>("/api/evaluation/alpha", {
      method: "POST",
      body: JSON.stringify({ model_run_id: modelRunId, window })
    }),
  runBacktest: (ticker: string, mode: string, modelRunId?: string) =>
    request<BacktestRunResponse>("/api/backtest/run", {
      method: "POST",
      body: JSON.stringify({ ticker, mode, model_run_id: modelRunId })
    }),
  backtestRuns: () => request<BacktestRunListResponse>("/api/backtest/runs"),
  trainRl: (ticker: string, algorithm = "PPO", timesteps = 50000, signalRunId?: string) =>
    request<SignalRunResponse>("/api/rl/train", {
      method: "POST",
      body: JSON.stringify({ ticker, algorithm, timesteps, signal_run_id: signalRunId || null })
    }),
  crossSectionUniverse: (limit = 300) => request<UniverseResponse>(`/api/cross-section/universe?limit=${limit}`),
  crossSectionLocalRank: (limit = 100) => request<CrossSectionRankResponse>(`/api/cross-section/local-rank?limit=${limit}`),
  crossSectionLocalBacktest: (limit = 100, longN = 10, shortN = 10) =>
    request<Record<string, unknown>>(`/api/cross-section/local-backtest?limit=${limit}&long_n=${longN}&short_n=${shortN}`, {
      method: "POST"
    }),
  crossSectionFactorReport: (limit = 100) => request<CrossSectionFactorReportResponse>(`/api/cross-section/factor-report?limit=${limit}`),
  factorSummary: (ticker: string) =>
    request<FeatureSummaryResponse>(`/api/factors/${encodeURIComponent(ticker)}/summary`, {
      method: "POST",
      body: JSON.stringify({ use_cache: true, horizon: 5, label_type: "regression" })
    }),
  factorAnalysis: (ticker: string) =>
    request<Record<string, unknown>>(`/api/factors/${encodeURIComponent(ticker)}/analysis`, {
      method: "POST",
      body: JSON.stringify({ use_cache: true, horizon: 5, label_type: "regression" })
    }),
  experiments: () => request<ExperimentListResponse>("/api/experiments"),
  compareExperiments: (expIds: string[]) =>
    request<ExperimentCompareResponse>("/api/experiments/compare", {
      method: "POST",
      body: JSON.stringify({ exp_ids: expIds })
    }),
  jobs: () => request<JobListResponse>("/api/jobs"),
  createPipelineJob: (ticker: string) =>
    request<JobResponse>("/api/jobs/pipeline", {
      method: "POST",
      body: JSON.stringify({ ticker, skip_rl: true })
    })
};
