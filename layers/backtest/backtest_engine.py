"""
回测引擎 — 支持三种模式对比

模式1 (signal_only): 直接按 signal_score 映射仓位
模式2 (signal_risk): signal_score → RiskManager → 仓位
模式3 (full): signal_score → RiskManager → RLExecutor → 仓位

生产级设计:
  - 持仓成本使用加权平均法（weighted average cost basis），每次买入更新、卖出减少
  - 已实现盈亏基于平均成本计算，未实现盈亏基于最新市价
  - 支持涨跌停限制接口（A 股可启用）
  - 支持停牌检测（成交量=0 时跳过）
  - 所有计算使用 float64 避免浮点误差累积
"""
import numpy as np
import pandas as pd
from layers.evaluation.metrics import MetricsCalculator
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BacktestConfig:
    """回测配置参数"""
    initial_balance: float = 10000.0
    commission: float = 0.001       # 手续费率
    slippage: float = 0.0005        # 滑点
    tax: float = 0.0                # 印花税（A 股卖出单边万分之5）
    max_position: float = 1.0       # 最大仓位
    max_single: float = 1.0         # 单资产上限
    # A 股特有
    enable_price_limits: bool = False   # 是否启用涨跌停限制
    price_limit_pct: float = 0.10      # 涨跌停幅度（主板 ±10%，科创/创业板 ±20%）
    # 执行模拟
    partial_fill: bool = False      # 是否模拟部分成交
    max_fill_ratio: float = 0.3     # 单次最多成交 ADV 比例
    # 报告
    save_trades: bool = True
    save_nav: bool = True


class BacktestEngine:
    """三种模式回测引擎"""

    def __init__(self, config):
        """
        Args:
            config: BacktestConfig 实例或兼容的命名元组
        """
        if isinstance(config, dict):
            self.config = BacktestConfig(**config)
        else:
            self.config = config

    def run(self, mode: str, data: pd.DataFrame,
            signal_scores: np.ndarray,
            risk_manager=None, rl_executor=None,
            prices: np.ndarray = None,
            volumes: np.ndarray = None,
            volatilities: np.ndarray = None,
            ) -> dict:
        """
        Args:
            mode: 'signal_only' | 'signal_risk' | 'full'
            data: test 集 DataFrame (含 OHLCV)
            signal_scores: 信号层输出的预测值, 长度 = len(data)
            risk_manager: RiskManager 实例 (mode=signal_risk/full 必需)
            rl_executor: RLExecutor 实例 (mode=full 必需)
            prices: 价格序列 (可选, 默认从 data['Close'] 读取)
            volumes: 成交量序列 (可选, 默认从 data['Volume'] 读取)
            volatilities: 波动率序列 (可选, 默认从 data 计算)

        Returns:
            {'metrics': dict, 'trades': list, 'nav': list, 'positions': list}
        """
        if mode not in ("signal_only", "signal_risk", "full"):
            raise ValueError(f"未知模式: {mode}")
        if mode in ("signal_risk", "full") and risk_manager is None:
            raise ValueError(f"模式 {mode} 需要 risk_manager")
        if mode == "full" and rl_executor is None:
            raise ValueError("模式 full 需要 rl_executor")

        n = len(data)
        cfg = self.config

        # ── 账户状态 ──
        balance = float(cfg.initial_balance)
        shares = 0.0
        # 加权平均成本（用于计算已实现盈亏）
        avg_cost = 0.0
        # 累计已实现盈亏
        realized_pnl = 0.0

        nav = [float(cfg.initial_balance)]
        positions = [0.0]
        trades = []
        stop_loss_triggered = False

        risk_manager.reset() if risk_manager else None

        # ── 预提取序列 ──
        close_arr = data['Close'].values.astype(np.float64)
        volume_arr = data['Volume'].values.astype(np.float64) if 'Volume' in data.columns else np.ones(n) * 1e6

        # 波动率: 优先使用外部传入, 否则从价格计算
        if volatilities is not None:
            vol_arr = np.asarray(volatilities, dtype=np.float64)
        else:
            ret = np.diff(close_arr) / close_arr[:-1]
            _v = np.full(n, 0.2)
            for i in range(20, n):
                _v[i] = np.std(ret[i - 20:i]) * np.sqrt(252)
            vol_arr = np.nan_to_num(_v, nan=0.2)

        for i in range(n):
            price = close_arr[i]
            vol = volume_arr[i]
            sig = float(signal_scores[i])

            # ── 停牌检测 ──
            if vol <= 0 or np.isnan(price) or price <= 0:
                nav.append(balance + shares * price if shares > 0 else balance)
                positions.append(shares * price / max(nav[-1], 1.0) if nav[-1] > 0 else 0.0)
                continue

            # ── A 股涨跌停限制 ──
            can_buy = True
            can_sell = True
            if cfg.enable_price_limits and i > 0:
                prev_close = close_arr[i - 1]
                upper_limit = prev_close * (1 + cfg.price_limit_pct)
                lower_limit = prev_close * (1 - cfg.price_limit_pct)
                if price >= upper_limit - 1e-8:
                    can_buy = False  # 涨停不能买入
                if price <= lower_limit + 1e-8 and shares > 0:
                    can_sell = False  # 跌停不能卖出

            # ── 当前状态 ──
            current_value = shares * price
            portfolio_value = balance + current_value
            current_pos = current_value / max(portfolio_value, 1.0)

            # 未实现盈亏（基于平均成本）
            if shares > 1e-8 and avg_cost > 0:
                unrealized_pnl = (price - avg_cost) / avg_cost
            else:
                unrealized_pnl = 0.0

            cur_vol = float(vol_arr[i])

            # ── 按模式决定目标仓位 ──
            if mode == "signal_only":
                target_pos = np.clip(sig, -cfg.max_position, cfg.max_position)
            elif mode == "signal_risk":
                result = risk_manager.compute_target_position(
                    signal_score=sig, current_price=price,
                    portfolio_value=portfolio_value,
                    current_volatility=cur_vol,
                    avg_daily_volume=vol,
                    current_position=current_pos,
                    unrealized_pnl=unrealized_pnl,
                )
                target_pos = result["target_position"]
                if result.get("is_stopped_out", False):
                    stop_loss_triggered = True
            elif mode == "full":
                result = risk_manager.compute_target_position(
                    signal_score=sig, current_price=price,
                    portfolio_value=portfolio_value,
                    current_volatility=cur_vol,
                    avg_daily_volume=vol,
                    current_position=current_pos,
                    unrealized_pnl=unrealized_pnl,
                )
                target_pos = result["target_position"]
                if result.get("is_stopped_out", False):
                    stop_loss_triggered = True

                # RL 推理
                obs = self._build_rl_obs(
                    sig, target_pos, current_pos,
                    unrealized_pnl, cur_vol,
                    close_arr[:i + 1], nav,
                )
                ratio = rl_executor.predict_execution_ratio(obs)
                target_pos = current_pos + ratio * (target_pos - current_pos)

            # ── 计算期望调仓量 ──
            target_value = target_pos * portfolio_value
            delta_value = target_value - current_value

            # ── 执行交易 ──
            if abs(delta_value) > 1e-6 * portfolio_value and not stop_loss_triggered:
                if delta_value > 0 and can_buy:
                    # 买入：现金限制 + 滑点 + 手续费
                    buy_value = min(delta_value, balance * (1 - 1e-6))
                    if buy_value > 0:
                        # 市场冲击调整
                        if cfg.partial_fill:
                            fill_ratio = min(1.0, vol * cfg.max_fill_ratio / (buy_value / price + 1e-8))
                            buy_value *= fill_ratio

                        shares_bought = buy_value / price
                        cost = buy_value * (cfg.commission + cfg.slippage)
                        tax_cost = buy_value * cfg.tax if cfg.tax > 0 else 0.0
                        total_cost = buy_value + cost + tax_cost

                        # 更新平均成本
                        old_cost_basis = shares * avg_cost
                        new_cost = shares_bought * price
                        shares += shares_bought
                        avg_cost = (old_cost_basis + new_cost) / shares if shares > 0 else price

                        balance -= total_cost

                        trades.append({
                            "step": i, "type": "buy", "price": price,
                            "shares": shares_bought, "cost": total_cost,
                            "delta": shares_bought / max(shares, 1e-6),
                            "avg_cost": avg_cost,
                        })

                elif delta_value < 0 and can_sell:
                    # 卖出
                    sell_value = min(abs(delta_value), current_value)
                    if sell_value > 0 and shares > 1e-8:
                        shares_sold = sell_value / price
                        shares_sold = min(shares_sold, shares)  # 防超卖

                        cost = sell_value * (cfg.commission + cfg.slippage)
                        tax_cost = sell_value * cfg.tax
                        revenue_net = sell_value - cost - tax_cost

                        # 已实现盈亏 = 卖出收入 - (卖出股数 × 平均成本)
                        cost_basis = shares_sold * avg_cost
                        trade_profit = revenue_net - cost_basis
                        trade_profit_pct = trade_profit / max(cost_basis, 1e-8)

                        realized_pnl += trade_profit

                        # 卖出后更新平均成本（不变，只是股数减少）
                        old_shares = shares
                        shares -= shares_sold
                        if shares <= 1e-8:
                            # 全部清仓
                            avg_cost = 0.0
                            shares = 0.0
                            trades.append({
                                "step": i, "type": "sell", "price": price,
                                "shares": shares_sold, "revenue": revenue_net,
                                "profit": trade_profit, "profit_pct": trade_profit_pct,
                                "delta": -1.0,
                                "realized_pnl": realized_pnl,
                            })
                        else:
                            trades.append({
                                "step": i, "type": "sell_partial", "price": price,
                                "shares": shares_sold, "revenue": revenue_net,
                                "profit": trade_profit, "profit_pct": trade_profit_pct,
                                "delta": -shares_sold / max(old_shares, 1e-8),
                                "realized_pnl": realized_pnl,
                            })

            # ── 日终净值 ──
            portfolio_value = balance + shares * price
            nav.append(portfolio_value)
            positions.append(shares * price / max(portfolio_value, 1.0) if portfolio_value > 0 else 0.0)

            # 止损清仓后重置标志
            if stop_loss_triggered and shares <= 1e-8:
                stop_loss_triggered = False

        # ── 强制平仓（回测结束时若有持仓） ──
        if shares > 1e-8:
            final_price = close_arr[-1]
            final_value = shares * final_price
            cost = final_value * (cfg.commission + cfg.slippage)
            tax_cost = final_value * cfg.tax
            balance += final_value - cost - tax_cost

            trade_profit = final_value - shares * avg_cost
            trade_profit_pct = trade_profit / max(shares * avg_cost, 1e-8)
            trades.append({
                "step": n - 1, "type": "force_close", "price": final_price,
                "shares": shares, "revenue": final_value - cost - tax_cost,
                "profit": trade_profit,
                "profit_pct": trade_profit_pct,
                "delta": -1.0,
                "realized_pnl": realized_pnl + trade_profit,
            })
            shares = 0.0

        nav[-1] = balance

        metrics = MetricsCalculator.compute_all(nav, trades)

        return {"metrics": metrics, "trades": trades, "nav": nav, "positions": positions}

    def _build_rl_obs(self, signal, target, pos, pnl, vol, price_hist, nav_hist) -> np.ndarray:
        """为 RL 单步推理构造近似观测向量 (40 维, 与 ExecutionEnv 对齐)"""
        obs = np.zeros(40, dtype=np.float32)

        # --- Core (0-8) ---
        obs[0] = signal                           # signal_score
        obs[1] = target                           # target_position
        obs[2] = pos                              # current_position
        obs[3] = pnl                              # unrealized_pnl
        obs[4] = vol                              # current_volatility
        dd = (max(nav_hist) - nav_hist[-1]) / max(max(nav_hist), 1e-6) if nav_hist else 0
        obs[5] = dd                               # drawdown
        cfg = self.config
        obs[6] = (cfg.commission + cfg.slippage) * abs(target - pos)  # turnover_cost
        obs[7] = 0.0                              # recent_action (保留)
        obs[8] = 1.0 if target != pos else 0.0    # need_trade flag

        # --- Regime (9-13, 从价格历史推导) ---
        if len(price_hist) >= 20:
            ret_20 = np.diff(price_hist[-20:]) / (price_hist[-21:-1] + 1e-8)
            obs[9] = np.sign(np.mean(ret_20))     # regime_trend
            obs[10] = np.std(ret_20) / (np.mean(np.abs(ret_20)) + 1e-8)  # regime_vol
            obs[11] = 0.0                         # regime_volume (保留)
            obs[12] = 0.0                         # regime_momentum (保留)
            obs[13] = obs[9]                      # regime_composite (简化)

        # --- History window (14-33: 4 steps × 5 features) ---
        hist_len = min(4, len(nav_hist) - 1)
        for i in range(hist_len):
            idx = len(nav_hist) - hist_len + i
            base = 14 + i * 5
            if idx > 0 and idx < len(nav_hist) and nav_hist[idx - 1] > 0:
                ret = nav_hist[idx] / nav_hist[idx - 1] - 1
                obs[base + 0] = 0.0              # hist_pos (保留)
                obs[base + 1] = 0.0              # hist_pnl (保留)
                obs[base + 2] = 0.0              # hist_action (保留)
                obs[base + 3] = ret              # hist_return
                obs[base + 4] = 0.0              # hist_cost (保留)

        # --- Auxiliary (34-39) ---
        mpos = cfg.max_position if hasattr(cfg, 'max_position') else 1.0
        obs[34] = min(1.0, abs(pos) / max(mpos, 0.01))  # position_utilization
        obs[35] = 0.0                                    # spread_estimate (保留)
        obs[36] = float(len(nav_hist)) / 252.0            # time_progress
        obs[37] = pnl / max(obs[34] * 10000, 1.0) if obs[34] > 0 else 0.0  # pnl_per_unit
        obs[38] = 0.0                                     # reserve
        obs[39] = 0.0                                     # reserve

        return obs
