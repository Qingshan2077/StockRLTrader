export interface SystemSummary {
  name: string;
  version: string;
  stage: string;
  data_dir: string;
  model_dir: string;
  log_dir: string;
  available_tickers: number;
}

export type SystemConfig = Record<string, unknown>;

export interface TickerInfo {
  ticker: string;
  custom_name?: string | null;
  rows: number;
  start?: string | null;
  end?: string | null;
  has_raw: boolean;
  has_processed: boolean;
}

export interface TickerListResponse {
  tickers: TickerInfo[];
}

export interface CandleRecord {
  date?: string;
  Date?: string;
  Open?: number;
  High?: number;
  Low?: number;
  Close?: number;
  Volume?: number;
  [key: string]: unknown;
}

export interface CandleResponse {
  ticker: string;
  source: string;
  row_count: number;
  columns: string[];
  records: CandleRecord[];
}

export interface DownloadResponse {
  ticker: string;
  success: boolean;
  rows: number;
  message: string;
}

export interface BatchDownloadResponse {
  results: DownloadResponse[];
}

export interface DeleteTickerResponse {
  ticker: string;
  deleted: string[];
  success: boolean;
}

export interface FeatureSummaryResponse {
  ticker: string;
  row_count: number;
  feature_count: number;
  label_count: number;
  cache_hit_rate: number;
  groups: string[];
  columns: string[];
  sample: Record<string, unknown>[];
}

export interface ExperimentResponse {
  exp_id: string;
  name: string;
  status?: string | null;
  tags: string[];
  config: Record<string, unknown>;
  final_metrics: Record<string, unknown>;
  created_at?: string | null;
  completed_at?: string | null;
}

export interface ExperimentListResponse {
  experiments: ExperimentResponse[];
}

export type ExperimentCompareResponse = Record<string, unknown>[];

export interface JobResponse {
  job_id: string;
  name: string;
  status: "queued" | "running" | "completed" | "failed" | "cancelled";
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  progress: number;
  message: string;
  error?: string | null;
}

export interface JobListResponse {
  jobs: JobResponse[];
}

export interface ProbabilityPredictionResponse {
  ticker: string;
  probabilities: Record<string, number>;
  metrics: Record<string, unknown>;
}

export interface AdvancedForecastResponse {
  ticker: string;
  forecast: Record<string, unknown>;
  trend: Record<string, unknown>;
  metrics: Record<string, unknown>;
}

export interface SignalRunResponse {
  run_id?: string | null;
  job_id?: string | null;
  status: string;
  result: Record<string, unknown>;
}

export interface SignalRunListResponse {
  runs: Record<string, unknown>[];
}

export interface BacktestRunResponse {
  job_id?: string | null;
  run_id?: string | null;
  status: string;
  result: Record<string, unknown>;
}

export interface BacktestRunListResponse {
  runs: Record<string, unknown>[];
}

export interface AlphaEvaluationResponse {
  run_id: string;
  ticker: string;
  report: Record<string, unknown>;
}

export interface UniverseResponse {
  market: string;
  tickers: string[];
  count: number;
}

export interface CrossSectionRankResponse {
  date?: string;
  count: number;
  model_status?: string;
  model_type?: string;
  feature_count?: number;
  message?: string;
  columns?: string[];
  records: Record<string, unknown>[];
}

export interface CrossSectionFactorReportResponse {
  rank_ic: Record<string, unknown>[];
  factor_columns: string[];
  factor_count?: number;
  message?: string;
}
