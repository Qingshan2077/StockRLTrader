import { object } from "./api/client";
import type { ExperimentDetail, Policy } from "./api/contracts";
export const statusNames: Record<string, string> = {
  queued: "等待执行",
  running: "执行中",
  cancelling: "正在取消",
  succeeded: "已完成",
  failed: "失败",
  cancelled: "已取消",
  interrupted: "已中断",
};
export const phaseNames: Record<string, string> = {
  preparing: "准备数据",
  training: "训练策略",
  validating: "验证检查点",
  evaluating: "评估策略与基准",
  publishing: "发布结果",
};
export const policyNames: Record<Policy, string> = {
  rl: "强化学习策略",
  cash: "全现金",
  buy_hold: "买入并持有",
  half: "固定半仓",
  trend: "趋势基准",
};
export const integrityNames: Record<string, string> = {
  pending: "等待结果",
  complete: "产物完整",
  partial: "产物不完整",
  corrupt: "产物损坏",
  unsupported: "不支持的版本",
};
export const metrics: Record<string, { label: string; percent?: boolean }> = {
  total_return: { label: "总收益", percent: true },
  annualized_return: { label: "年化收益", percent: true },
  volatility: { label: "年化波动", percent: true },
  sharpe: { label: "夏普比率" },
  max_drawdown: { label: "最大回撤", percent: true },
  total_turnover: { label: "总换手" },
  total_cost: { label: "总交易成本" },
  average_weight: { label: "平均仓位", percent: true },
  trade_count: { label: "交易次数" },
};
export function number(value: unknown, digits = 3): string {
  return typeof value === "number" && Number.isFinite(value)
    ? value.toLocaleString("zh-CN", { maximumFractionDigits: digits })
    : "未定义";
}
export function metric(key: string, value: unknown): string {
  return typeof value === "number" && Number.isFinite(value)
    ? metrics[key]?.percent
      ? `${number(value * 100, 2)}%`
      : number(value)
    : "未定义";
}
export function timestamp(value: string | null): string {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? value
    : date.toLocaleString("zh-CN", { hour12: false });
}
export function experimentName(item: ExperimentDetail) {
  return item.data_label || item.dataset?.display_name || "来源未标注";
}
export function algorithm(item: ExperimentDetail) {
  return item.request?.algorithm ?? item.legacy_config?.algorithm ?? "未知";
}
export function runMetrics(
  run: Record<string, unknown>,
  policy: Policy,
): Record<string, unknown> {
  return policy === "rl"
    ? object(run.metrics)
    : object(object(object(run.baselines)[policy]).metrics);
}
export const differenceNames: Record<string, string> = {
  data: "数据快照",
  dataset_id: "数据集",
  snapshot_sha256: "数据指纹",
  filtered_data_sha256: "筛选数据指纹",
  splits: "日期切分",
  trading_config: "交易与奖励配置",
  algorithm: "算法",
  timesteps: "训练步数",
  seeds: "随机种子",
  episode_length: "训练片段长度",
  train_ratio: "训练集比例",
  val_ratio: "验证集比例",
  versions: "语义版本",
  data_label: "数据标注",
  is_synthetic: "合成数据标记",
  purpose: "实验用途",
  core_semantics_version: "核心语义版本",
  metrics_version: "指标版本",
  artifact_schema_version: "产物版本",
};
