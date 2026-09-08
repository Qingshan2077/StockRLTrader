// Generated from Python annotations by scripts/generate_contracts.py. Do not edit.
// All response fields are present; request defaults come from /capabilities.
// Runtime OpenAPI parity remains a separate integration acceptance check.

export type JobStatus = "queued" | "running" | "cancelling" | "succeeded" | "failed" | "cancelled" | "interrupted";
export type JobKind = "train" | "replay";
export type Phase = "preparing" | "training" | "validating" | "evaluating" | "publishing";
export type Integrity = "pending" | "complete" | "partial" | "corrupt" | "unsupported";
export type Seed = number;
export interface TradingConfigModel {
  initial_cash: number;
  commission: number;
  slippage: number;
  sell_tax: number;
  min_commission: number;
  lot_size: number;
  max_participation: number;
  t_plus_one: boolean;
  drawdown_penalty: number;
  turnover_penalty: number;
}

export interface ExperimentRequest {
  dataset_id: string;
  purpose: "technical_validation" | "research";
  research_question: string;
  comparison_notes: string;
  start_date: string | null;
  end_date: string | null;
  algorithm: "PPO" | "SAC";
  timesteps: number;
  seeds: Array<Seed>;
  episode_length: number | null;
  train_ratio: number;
  val_ratio: number;
  trading_config: TradingConfigModel;
}

export interface DatasetMetadata {
  display_name: string | null;
  adjustment: string;
  quote_unit: string;
  source_description: string;
}

export interface DemoDatasetRequest extends DatasetMetadata {
  rows: number;
  seed: Seed;
}

export interface LocalDatasetRequest extends DatasetMetadata {
  source_id: string;
}

export interface LocalSource {
  source_id: string;
  display_name: string;
}

export interface Dataset {
  dataset_id: string;
  source_kind: "csv" | "demo" | "local";
  display_name: string;
  source_description: string;
  is_synthetic: boolean;
  adjustment: string;
  quote_unit: string;
  rows: number;
  first_date: string;
  last_date: string;
  snapshot_sha256: string;
  original_sha256: string | null;
  created_at: string;
}

export interface BarRow {
  Date: string;
  Open: number;
  High: number;
  Low: number;
  Close: number;
  Volume: number;
}

export interface DatasetPreview {
  dataset: Dataset;
  total_rows: number;
  sample: Array<BarRow>;
  trainable: boolean;
  warnings: Array<string>;
}

export interface SplitInterval {
  start: number;
  end: number;
  reward_count: number;
  observation_start_date: string;
  first_reward_date: string;
  last_reward_date: string;
}

export interface ExperimentPreview {
  canonical_request: ExperimentRequest;
  request_sha256: string;
  filtered_data_sha256: string;
  splits: Record<string, SplitInterval>;
  seed_count: number;
  warnings: Array<string>;
}

export interface SubmissionResult {
  experiment_id: string;
  job_id: string;
}

export interface ErrorSummary {
  code: string;
  message: string;
  details: Array<Record<string, unknown>> | Record<string, unknown> | null;
}

export interface JobDetail {
  job_id: string;
  experiment_id: string;
  kind: JobKind;
  status: JobStatus;
  revision: number;
  phase: Phase | null;
  seed: number | null;
  seed_index: number | null;
  seed_count: number;
  requested_steps_total: number;
  actual_steps_total: number;
  training_fraction: number | null;
  completed_seeds: number;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  heartbeat_at: string | null;
  worker_available: boolean;
  recovery_waiting: boolean;
  error: ErrorSummary | null;
}

export interface StateEventPayload {
  status: JobStatus;
  revision: number;
}

export interface PhaseEventPayload {
  phase: Phase;
  seed: number | null;
  seed_index: number | null;
}

export interface ProgressEventPayload {
  seed: number | null;
  seed_index: number | null;
  actual_steps_total: number;
  requested_steps_total: number;
  completed_seeds: number;
  training_fraction: number | null;
}

export interface JobEvent {
  job_id: string;
  seq: number;
  occurred_at: string;
  event_type: "state" | "phase" | "progress" | "error";
  payload: StateEventPayload | PhaseEventPayload | ProgressEventPayload | ErrorSummary;
}

export interface EventPage {
  items: Array<JobEvent>;
  next_seq: number;
  has_more: boolean;
}

export interface Page<T> {
  items: Array<T>;
  next_cursor: string | null;
  has_more: boolean;
}

export interface Versions {
  artifact_schema_version: number | "legacy-v0" | "unknown";
  core_semantics_version: number | "unknown";
  metrics_version: number | "unknown";
}

export interface ArtifactRecord {
  artifact_id: string;
  experiment_id: string;
  kind: string;
  filename: string;
  size: number;
  sha256: string;
  download_url: string;
}

export interface LegacyConfiguration {
  algorithm: "PPO" | "SAC" | null;
  trading_config: TradingConfigModel | null;
  timesteps: number | null;
  seeds: Array<Seed>;
  episode_length: number | null;
  train_ratio: number | null;
  val_ratio: number | null;
}

export interface ExperimentDetail {
  experiment_id: string;
  kind: JobKind;
  source_experiment_id: string | null;
  job_id: string | null;
  run_id: string | null;
  created_at: string;
  request: ExperimentRequest | null;
  request_sha256: string | null;
  filtered_data_sha256: string | null;
  legacy_config: LegacyConfiguration | null;
  dataset: Dataset | null;
  data_label: string | null;
  is_synthetic: boolean | null;
  splits: Record<string, SplitInterval>;
  versions: Versions;
  integrity: Integrity;
  replayable: boolean;
  replay_block_reason: string | null;
  runs: Array<Record<string, unknown>>;
  aggregate: Record<string, unknown> | null;
  baseline_aggregate: Record<string, Record<string, unknown>>;
  artifacts: Array<ArtifactRecord>;
}

export interface Capabilities {
  api_version: number;
  database_schema_version: number;
  artifact_schema_version: number;
  core_semantics_version: number;
  metrics_version: number;
  algorithms: Array<string>;
  policies: Array<string>;
  defaults: Record<string, unknown>;
  limits: Record<string, unknown>;
  worker_available: boolean;
}

export type Policy = "rl" | "cash" | "buy_hold" | "half" | "trend";
export interface ErrorEnvelope {
  error: ErrorSummary;
  request_id: string;
}

export interface SeriesPoint {
  date: string;
  nav: number;
  cash: number;
  shares: number;
  weight: number;
  requested_weight: number | null;
  executed_weight: number | null;
  turnover: number;
  cost: number;
}

export interface SeriesResponse {
  seed: number;
  policy: Policy;
  points: Array<SeriesPoint>;
  original_count: number;
  returned_count: number;
  is_sampled: boolean;
}

export interface TradeRow {
  date: string;
  side: "buy" | "sell";
  shares: number;
  price: number;
  cost: number;
}

export interface TradesResponse {
  seed: number;
  policy: Policy;
  items: Array<TradeRow>;
  total_rows: number;
  next_cursor: string | null;
  has_more: boolean;
}

export interface ComparisonResponse {
  experiments: Array<ExperimentDetail>;
  same_conditions: boolean;
  differences: Array<Record<string, unknown>>;
  notes: Array<string>;
}

export interface EmptyRequest {
}
