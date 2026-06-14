"""
Risk Engine — 组合全部约束, 输出 adjusted_position + risk_report

执行顺序:
  1. 仓位约束 (max gross/net/single)
  2. 止损检查 (固定止损 + 重新入场惩罚)
  3. 波动率约束 (目标波动率缩放)
  4. 回撤控制 (熔断 + 动态降仓)
  5. 流动性约束 (ADV 限制)
  6. 换手率限制 (单日最大调仓比例)
  7. 成本估算 (佣金 + 滑点 + 市场冲击)

生产级设计:
  - 每个约束的结果可独立追踪
  - 风控报告包含每层约束是否触发的 flag
  - 支持序列化 reset（多模式回测时使用）
"""
import numpy as np
from layers.risk.constraints import (
    PositionConstraints, VolatilityConstraints,
    DrawdownControl, LiquidityConstraints,
)
from layers.risk.cost_model import CostModel


class RiskEngine:
    """统一风险引擎 — 按顺序执行所有约束"""

    def __init__(self, config_loader=None):
        # 从 YAML 加载参数, 或使用默认值
        risk_cfg = config_loader.get("risk", {}) if config_loader else {}
        pos_cfg = risk_cfg.get("position", {})
        vol_cfg = risk_cfg.get("volatility", {})
        dd_cfg = risk_cfg.get("drawdown", {})
        liq_cfg = risk_cfg.get("liquidity", {})
        to_cfg = risk_cfg.get("turnover", {})

        self.position = PositionConstraints(
            max_gross=pos_cfg.get("max_gross_exposure", 1.0),
            max_net=pos_cfg.get("max_net_exposure", 1.0),
            max_single=pos_cfg.get("max_single_asset_weight", 1.0),
        )
        self.volatility = VolatilityConstraints(
            target_vol=vol_cfg.get("target_vol", 0.25),
            vol_window=vol_cfg.get("vol_window", 20),
            vol_threshold=vol_cfg.get("vol_threshold", 1.5),
            enabled=vol_cfg.get("enabled", True),
        )
        self.drawdown = DrawdownControl(
            circuit_breaker=dd_cfg.get("circuit_breaker", -0.20),
            deleverage_factor=dd_cfg.get("deleverage_factor", 0.5),
            enabled=dd_cfg.get("enabled", True),
        )
        self.liquidity = LiquidityConstraints(
            adv_limit_ratio=liq_cfg.get("adv_limit_ratio", 0.01),
            minimum_volume=liq_cfg.get("minimum_volume", 10000),
            enabled=liq_cfg.get("enabled", True),
        )

        self.stop_loss_threshold = risk_cfg.get("stop_loss", {}).get("threshold", -0.05)
        self.reentry_penalty = risk_cfg.get("stop_loss", {}).get("reentry_penalty", 0.5)
        self.max_turnover = to_cfg.get("max_daily_turnover", 0.3)

        bk_cfg = config_loader.get("backtest", {}) if config_loader else {}
        impact_cfg = bk_cfg.get("market_impact", {})
        self.cost_model = CostModel(
            commission=bk_cfg.get("commission", 0.001),
            slippage=bk_cfg.get("slippage", 0.0005),
            tax=bk_cfg.get("tax", 0.0),
            impact_model=impact_cfg.get("model", "sqrt"),
            k=impact_cfg.get("k", 0.1),
        )

        # 内部状态
        self._stop_loss_triggered = False
        self._prev_position = 0.0
        self._peak_value = 0.0  # 追踪峰值净值用于动态回撤

    def process(self, signal_score: float, current_price: float,
                portfolio_value: float, current_volatility: float,
                avg_daily_volume: float, current_position: float = 0.0,
                unrealized_pnl: float = 0.0) -> dict:
        """
        执行全部风控约束流水线

        Returns:
            target_position: float           — 原始信号映射后的目标仓位
            adjusted_position: float         — 经过全部约束后的最终仓位
            is_stopped_out: bool             — 是否触发止损
            constraints_log: list[str]       — 每层约束的日志
            cost_estimate: float             — 估计交易成本
            risk_report: dict                — 每层约束的触发 flag
        """
        log = []
        constraint_flags = {}

        # ── 0. 记录原始信号 ──
        raw_signal = float(signal_score)

        # ── 1. 仓位约束 ──
        raw_pos = np.clip(signal_score, -1.0, 1.0)
        pos, pos_log = self.position.apply(raw_pos)
        if pos_log:
            log.extend(pos_log.values())
        position_was_limited = abs(pos) < abs(raw_pos) - 1e-8
        constraint_flags["position_limited"] = bool(position_was_limited)

        # ── 2. 止损检查 ──
        stopped = False
        if current_position != 0 and unrealized_pnl <= self.stop_loss_threshold:
            self._stop_loss_triggered = True
            stopped = True
            pos = 0.0
            log.append(f"stop_loss: pnl={unrealized_pnl:.4f} ≤ {self.stop_loss_threshold}")
        elif self._stop_loss_triggered and stopped == False:
            # 重新入场减半惩罚
            pos *= self.reentry_penalty
            self._stop_loss_triggered = False
            log.append(f"reentry: position halved to {pos:.4f}")
        constraint_flags["stop_loss_triggered"] = stopped

        # ── 3. 波动率约束 ──
        pos, vol_log = self.volatility.apply(pos, current_volatility)
        if vol_log:
            log.extend(vol_log.values())
        constraint_flags["volatility_reduced"] = bool(vol_log)

        # ── 4. 回撤控制 ──
        self._peak_value = max(self._peak_value, portfolio_value)
        current_drawdown = (portfolio_value - self._peak_value) / max(self._peak_value, 1.0)
        pos, dd_log = self.drawdown.apply(pos, current_drawdown, current_position)
        if dd_log:
            log.extend(dd_log.values())
        constraint_flags["drawdown_controlled"] = bool(dd_log)

        # ── 5. 流动性约束 ──
        pos, liq_log = self.liquidity.apply(pos, avg_daily_volume,
                                             current_price, portfolio_value)
        if liq_log:
            log.extend(liq_log.values())
        constraint_flags["liquidity_filtered"] = bool(liq_log)

        # ── 6. 换手率限制 ──
        delta = pos - current_position
        max_delta = self.max_turnover
        if abs(delta) > max_delta:
            delta = np.clip(delta, -max_delta, max_delta)
            pos = current_position + delta
            log.append(f"turnover_limit: delta clipped from {delta:.4f} to {np.clip(pos - current_position, -max_delta, max_delta):.4f}")
        constraint_flags["turnover_limited"] = abs(pos - current_position) < abs(raw_pos - current_position) - 1e-8

        # ── 7. 成本估算 ──
        turnover_value = abs(pos - current_position) * portfolio_value
        adv = avg_daily_volume * current_price if avg_daily_volume > 0 else portfolio_value
        cost_detail = self.cost_model.estimate(turnover_value, adv, current_price)

        risk_report = {
            "raw_signal": raw_signal,
            "raw_position": float(raw_pos),
            "adjusted_position": float(pos),
            **constraint_flags,
            **cost_detail,
        }

        self._prev_position = pos

        return {
            "target_position": float(raw_pos),
            "adjusted_position": float(pos),
            "is_stopped_out": stopped,
            "constraints_log": log,
            "cost_estimate": cost_detail.get("total", 0.0),
            "risk_report": risk_report,
        }

    def compute_target_position(self, signal_score: float, current_price: float,
                                 portfolio_value: float, current_volatility: float,
                                 avg_daily_volume: float, current_position: float = 0.0,
                                 unrealized_pnl: float = 0.0) -> dict:
        """兼容旧 RiskManager 接口 — 调用 process() 并返回 target_position / is_stopped_out"""
        result = self.process(signal_score, current_price, portfolio_value,
                              current_volatility, avg_daily_volume,
                              current_position, unrealized_pnl)
        # BacktestEngine 使用 adjusted_position 作为实际的 target
        result["target_position"] = result["adjusted_position"]
        return result

    def reset(self) -> None:
        """重置内部状态（开始新回测前调用）"""
        self._stop_loss_triggered = False
        self._prev_position = 0.0
        self._peak_value = 0.0
        self.drawdown.reset()

    def __repr__(self) -> str:
        return (f"RiskEngine(stop_loss={self.stop_loss_threshold}, "
                f"max_turnover={self.max_turnover})")
