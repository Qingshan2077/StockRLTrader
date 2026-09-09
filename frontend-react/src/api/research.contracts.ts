// Generated from stockrl/research/contracts.py. Do not edit.

export interface TrainingBudget {
  requested_timesteps: number;
  episode_length: number;
  algorithm: "PPO" | "SAC";
}

export interface AcceptanceThresholds {
  minimum_folds: 4;
  required_seeds: Array<number>;
  comparable_fraction: 0.8;
  winning_fold_fraction: 0.6;
  maximum_drawdown_difference: 0.02;
  risk_absolute_tolerance: 0.02;
  risk_relative_tolerance: 0.2;
  minimum_assets: 5;
  winning_asset_fraction: 0.6;
}

export interface SessionWindow {
  calendar_start: string;
  calendar_end: string;
  start_session: string;
  end_session: string;
  initial_session: string;
  reward_sessions: Array<string>;
  warmup_sessions: Array<string>;
}

export interface FoldPlan {
  fold_id: string;
  instrument_id: string | null;
  train: SessionWindow;
  validation: SessionWindow;
  test: SessionWindow;
}

export interface ResearchProtocolDraft {
  protocol_id: string;
  version: 2;
  hypothesis: string;
  scope: "single_asset" | "asset_set";
  instrument_ids: Array<string>;
  dataset_ids: Array<string>;
  dataset_fingerprints: Record<string, string>;
  market_profile_ids: Array<string>;
  market_profile_fingerprints: Record<string, string>;
  asset_selection_note: string;
  first_test_session: string;
  fold_count: number;
  seeds: Array<number>;
  training_budget: TrainingBudget;
  checkpoint_budget: number;
  primary_reference_ids: Array<"fixed_50" | "matched_fixed">;
  cost_scenarios: Array<"base" | "execution_x2" | "execution_x3">;
  qualification: "exploratory" | "declared_holdout";
  acceptance_thresholds: AcceptanceThresholds;
}

export interface ResearchProtocol extends ResearchProtocolDraft {
  fold_plan: Array<FoldPlan>;
  checkpoint_steps: Array<number>;
}

export interface LockedProtocol {
  canonical_json: string;
  sha256: string;
}

export interface BudgetPreview {
  unit_count: number;
  requested_total_steps: number;
  rollout_upper_bound: number;
  actual_steps_per_unit: number;
  rollout_steps: number;
  checkpoint_steps: Array<number>;
  checkpoint_evaluations: number;
  calibration_evaluations: number;
  test_evaluations: number;
  extra_evaluations: number;
}

export interface ResearchPreview {
  fold_plan: Array<FoldPlan>;
  budget: BudgetPreview;
  checkpoint_steps: Array<number>;
  qualification: "exploratory" | "declared_holdout";
  blockers: Array<string>;
  omitted_folds: Array<string>;
}

export interface UnitKey {
  instrument_id: string;
  fold_id: string;
  seed: number;
}

export interface StrategyMetrics {
  net_return: number;
  cagr: number | null;
  annualized_volatility: number | null;
  sharpe: number | null;
  max_drawdown: number;
  fees: number;
  turnover: number;
  average_exposure: number;
  trade_count: number;
  reward_sessions: number;
  final_nav: number | null;
  final_receivables: number;
}

export interface CheckpointResult {
  checkpoint_id: string;
  actual_steps: number;
  gradient_updates: number;
  model_sha256: string;
  validation_log_return: number;
  evaluated_at: string;
  metrics: StrategyMetrics;
  validation_reward: number | null;
}

export interface CostResult {
  scenario_id: "base" | "execution_x2" | "execution_x3";
  model_sha256: string;
  metrics: Record<string, StrategyMetrics>;
  risk_comparable: boolean | null;
  artifact_paths: Record<string, string>;
}

export interface UnitResult {
  key: UnitKey;
  status: "completed" | "failed" | "cancelled" | "pending" | "running";
  cost_results: Array<CostResult>;
  candidates: Array<CheckpointResult>;
  selected_checkpoint_id: string | null;
  selection_reason: string | null;
  matched_weight: number | null;
  calibration_error: number | null;
  actual_steps: number;
  fingerprints_valid: boolean;
  rule_coverage_complete: boolean;
  error: string | null;
}

export interface ExposureRecord {
  instrument_id: string;
  start_session: string;
  end_session: string;
  scope: "whole_period_knowledge" | "evaluation" | "exposed_test";
  reason: string;
  recorded_at: string;
  protocol_id: string | null;
  source: string;
}

export interface ResearchReport {
  protocol_id: string;
  technical_status: "completed" | "failed" | "cancelled" | "incomplete";
  qualification: "exploratory" | "declared_holdout";
  economic_outcome: "insufficient_evidence" | "no_added_value" | "candidate_edge";
  provisional_outcome: "insufficient_evidence" | "no_added_value" | "candidate_edge" | null;
  reasons: Array<string>;
  assets: Record<string, unknown>;
  units: Array<UnitResult>;
}

export interface DiagnosticsReport {
  boundary_action_ratio: number | null;
  clipping_ratio: number | null;
  near_half_ratio: number | null;
  warnings: Array<string>;
  unavailable: Array<string>;
  feature_stats: Record<string, unknown>;
  regimes: Record<string, unknown>;
  training_log: Array<Record<string, unknown>>;
  summary: Record<string, unknown>;
}
