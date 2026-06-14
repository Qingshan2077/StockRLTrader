"""
截面组合回测引擎 — 多空组合回测

核心功能:
  1. 每日按预测排名选股 (Top K 做多, Bottom K 做空)
  2. 等权 / 市值加权 / 波动率加权
  3. 行业约束 (单行业持仓上限)
  4. 换手率追踪
  5. 交易成本模拟
  6. 输出完整净值曲线 + 绩效报告

用法:
    engine = CrossSectionalBacktest(long_n=30, short_n=30)
    result = engine.run(panel_with_predictions, return_col="future_return")
"""

import numpy as np
import pandas as pd
from typing import Optional, Literal
from dataclasses import dataclass, field


@dataclass
class CSBacktestResult:
    """截面回测结果"""
    nav: pd.Series                        # 每日净值 (初始=1.0)
    benchmark: pd.Series                  # 基准净值
    long_nav: pd.Series                   # 多头端净值
    short_nav: pd.Series                  # 空头端净值
    positions: pd.DataFrame               # 每日持仓 (date × code, weight)
    turnover: pd.Series                   # 每日换手率
    metrics: dict                         # 绩效指标
    daily_returns: pd.Series              # 每日收益
    long_returns: pd.Series               # 多头每日收益
    short_returns: pd.Series              # 空头每日收益


class CrossSectionalBacktest:
    """截面多空组合回测引擎"""

    def __init__(
        self,
        long_n: int = 30,
        short_n: int = 30,
        weight_scheme: Literal["equal", "mcap", "vol"] = "equal",
        commission: float = 0.0003,
        slippage: float = 0.0005,
        max_sector_weight: float = 0.3,
        max_single_weight: float = 0.05,
        max_turnover: float = 0.5,
        verbose: bool = True,
    ):
        """
        Args:
            long_n: 做多股票数量
            short_n: 做空股票数量 (0 则为纯多)
            weight_scheme: 权重方案
            commission: 手续费率 (双边)
            slippage: 滑点
            max_sector_weight: 单行业最大权重
            max_single_weight: 单只股票最大权重
            max_turnover: 每日最大换手率
            verbose: 打印信息
        """
        self.long_n = long_n
        self.short_n = short_n
        self.weight_scheme = weight_scheme
        self.commission = commission
        self.slippage = slippage
        self.max_sector_weight = max_sector_weight
        self.max_single_weight = max_single_weight
        self.max_turnover = max_turnover
        self.verbose = verbose

    def run(
        self,
        panel: pd.DataFrame,
        rank_col: str = "rank_pct",
        return_col: str = "future_return",
        sector_col: str = "sector",
        mcap_col: Optional[str] = None,
        vol_col: Optional[str] = None,
        benchmark_col: Optional[str] = None,
        initial_cash: float = 1_000_000,
    ) -> CSBacktestResult:
        """运行截面多空回测

        Args:
            panel: MultiIndex (date, code) 面板, 必须含 rank_col + return_col
            rank_col: 排名依据列 [0, 1], 越高越好
            return_col: 未来收益列
            sector_col: 行业列
            mcap_col: 市值列 (可选, 用于市值加权)
            vol_col: 波动率列 (可选, 用于波动率倒数加权)
            benchmark_col: 基准收益列 (可选, 计算超额)
            initial_cash: 初始资金

        Returns:
            CSBacktestResult
        """
        dates = panel.index.get_level_values("date").unique().sort_values()
        n_dates = len(dates)
        if n_dates < 2:
            raise ValueError(f"数据不足: {n_dates} 个日期")

        # ── 准备存储 ──
        nav = np.ones(n_dates)
        long_nav = np.ones(n_dates)
        short_nav = np.ones(n_dates)
        daily_ret = np.zeros(n_dates)
        long_ret = np.zeros(n_dates)
        short_ret = np.zeros(n_dates)
        turnover = np.zeros(n_dates)
        prev_long_positions: set[str] = set()
        prev_short_positions: set[str] = set()
        all_positions: list[pd.Series] = []
        bench_returns = np.zeros(n_dates)

        for i in range(1, n_dates):
            dt = dates[i]
            dt_prev = dates[i - 1]
            chunk = panel.loc[dt].copy()
            prev_chunk = panel.loc[dt_prev] if dt_prev in panel.index else None

            # ── 选股 ──
            valid = chunk[rank_col].dropna()
            sorted_codes = valid.sort_values(ascending=False)
            long_codes = set(sorted_codes.head(self.long_n).index)
            short_codes = set(sorted_codes.tail(self.short_n).index) if self.short_n > 0 else set()

            # 移除重叠
            overlap = long_codes & short_codes
            long_codes -= overlap
            short_codes -= overlap

            # ── 权重计算 ──
            long_weights = self._compute_weights(
                chunk, long_codes, sector_col, mcap_col, vol_col
            )
            short_weights = self._compute_weights(
                chunk, short_codes, sector_col, mcap_col, vol_col, negative=True
            )

            # ── 换手率 ──
            curr_long = set(long_weights.index)
            curr_short = set(short_weights.index)
            long_turn = len(curr_long - prev_long_positions) / max(len(curr_long), 1)
            short_turn = len(curr_short - prev_short_positions) / max(len(curr_short), 1)
            turnover[i] = (long_turn + short_turn) / 2

            # ── 收益计算 ──
            if return_col in chunk.columns:
                # 多头收益
                l_ret = self._portfolio_return(chunk, long_weights, return_col)
                # 空头收益 (做空 = 负收益)
                s_ret = self._portfolio_return(chunk, short_weights, return_col)
            else:
                l_ret = s_ret = 0.0

            # 扣除交易成本 (基于换手)
            tc = (long_turn + short_turn) * (self.commission + self.slippage)

            long_ret[i] = l_ret
            short_ret[i] = -s_ret  # 做空: 下跌=正收益
            daily_ret[i] = l_ret - s_ret - tc
            nav[i] = nav[i - 1] * (1 + daily_ret[i]) if not np.isnan(daily_ret[i]) else nav[i - 1]
            long_nav[i] = long_nav[i - 1] * (1 + l_ret) if not np.isnan(l_ret) else long_nav[i - 1]
            short_nav[i] = short_nav[i - 1] * (1 - s_ret) if not np.isnan(s_ret) else short_nav[i - 1]

            # ── 记录持仓 ──
            pos = pd.concat([long_weights, short_weights])
            pos.name = dt
            all_positions.append(pos)

            prev_long_positions = curr_long
            prev_short_positions = curr_short

            if benchmark_col and benchmark_col in chunk.columns:
                bench_returns[i] = chunk[benchmark_col].mean()

        # ── 构建结果 ──
        nav_series = pd.Series(nav, index=dates, name="nav")
        long_nav_s = pd.Series(long_nav, index=dates, name="long_nav")
        short_nav_s = pd.Series(short_nav, index=dates, name="short_nav")
        daily_ret_s = pd.Series(daily_ret, index=dates, name="daily_return")
        long_ret_s = pd.Series(long_ret, index=dates, name="long_return")
        short_ret_s = pd.Series(short_ret, index=dates, name="short_return")
        turnover_s = pd.Series(turnover, index=dates, name="turnover")

        # 基准
        bench_nav = pd.Series(np.cumprod(1 + bench_returns), index=dates, name="benchmark")

        # 持仓 DataFrame
        pos_df = pd.DataFrame(all_positions).T if all_positions else pd.DataFrame()

        # ── 绩效指标 ──
        metrics = self._compute_metrics(daily_ret_s, bench_returns, nav_series, turnover_s)

        if self.verbose:
            self._print_summary(metrics)

        return CSBacktestResult(
            nav=nav_series,
            benchmark=bench_nav,
            long_nav=long_nav_s,
            short_nav=short_nav_s,
            positions=pos_df,
            turnover=turnover_s,
            metrics=metrics,
            daily_returns=daily_ret_s,
            long_returns=long_ret_s,
            short_returns=short_ret_s,
        )

    def _compute_weights(
        self,
        chunk: pd.DataFrame,
        codes: set[str],
        sector_col: str,
        mcap_col: Optional[str],
        vol_col: Optional[str],
        negative: bool = False,
    ) -> pd.Series:
        """计算权重"""
        if not codes:
            return pd.Series(dtype=float)

        if self.weight_scheme == "mcap" and mcap_col and mcap_col in chunk.columns:
            weights = chunk.loc[list(codes), mcap_col].abs()
        elif self.weight_scheme == "vol" and vol_col and vol_col in chunk.columns:
            inv_vol = 1 / chunk.loc[list(codes), vol_col].clip(lower=1e-6)
            weights = inv_vol
        else:
            weights = pd.Series(1.0, index=list(codes))

        # 行业约束
        if sector_col in chunk.columns:
            sectors = chunk.loc[list(codes), sector_col]
            for sec in sectors.unique():
                sec_mask = sectors == sec
                total = weights[sec_mask].sum()
                if total > self.max_sector_weight:
                    weights[sec_mask] *= self.max_sector_weight / total

        # 单只股票约束
        weights = weights.clip(upper=self.max_single_weight)

        # 归一化
        total = weights.sum()
        if total > 0:
            weights = weights / total

        if negative:
            weights = -weights

        return weights

    def _portfolio_return(
        self,
        chunk: pd.DataFrame,
        weights: pd.Series,
        return_col: str,
    ) -> float:
        """计算组合收益"""
        if weights.empty:
            return 0.0
        common = weights.index.intersection(chunk.index)
        if common.empty:
            return 0.0
        rets = chunk.loc[common, return_col]
        w = weights[common]
        return (w * rets).sum()

    def _compute_metrics(
        self,
        daily_ret: pd.Series,
        bench_ret: np.ndarray,
        nav: pd.Series,
        turnover: pd.Series,
    ) -> dict:
        """计算绩效指标"""
        metrics = {}

        # 基本指标
        total_days = len(daily_ret)
        ann_factor = np.sqrt(252 if total_days > 252 else 252)

        metrics["total_return"] = float(nav.iloc[-1] - 1)
        metrics["annualized_return"] = float(nav.iloc[-1] ** (252 / total_days) - 1)
        metrics["annualized_vol"] = float(daily_ret.std() * ann_factor)
        metrics["sharpe_ratio"] = float(
            daily_ret.mean() / daily_ret.std() * ann_factor
        ) if daily_ret.std() > 0 else 0.0
        metrics["max_drawdown"] = float(self._max_drawdown(nav))
        metrics["calmar_ratio"] = float(
            metrics["annualized_return"] / abs(metrics["max_drawdown"])
        ) if metrics["max_drawdown"] < 0 else 0.0

        # 换手率
        metrics["avg_turnover"] = float(turnover.mean())
        metrics["avg_turnover_daily"] = float(turnover.mean())

        # 胜率
        metrics["win_rate"] = float((daily_ret > 0).mean())
        metrics["avg_win"] = float(daily_ret[daily_ret > 0].mean()) if (daily_ret > 0).any() else 0.0
        metrics["avg_loss"] = float(daily_ret[daily_ret < 0].mean()) if (daily_ret < 0).any() else 0.0

        # 超额
        total_bench = bench_ret[-1] if len(bench_ret) > 0 else 0
        metrics["benchmark_return"] = float(total_bench)
        metrics["excess_return"] = float(metrics["total_return"] - total_bench)

        # 三分位
        metrics["profit_days"] = int((daily_ret > 0).sum())
        metrics["loss_days"] = int((daily_ret < 0).sum())

        return metrics

    def _max_drawdown(self, nav: pd.Series) -> float:
        """计算最大回撤"""
        peak = nav.expanding().max()
        dd = (nav - peak) / peak
        return float(dd.min())

    def _print_summary(self, metrics: dict) -> None:
        print(f"\n{'='*50}")
        print(f"📊 截面多空回测结果")
        print(f"{'='*50}")
        print(f"  总收益:      {metrics['total_return']*100:+.2f}%")
        print(f"  年化收益:    {metrics['annualized_return']*100:+.2f}%")
        print(f"  年化波动:    {metrics['annualized_vol']*100:.2f}%")
        print(f"  夏普比率:    {metrics['sharpe_ratio']:.2f}")
        print(f"  最大回撤:    {metrics['max_drawdown']*100:.2f}%")
        print(f"  卡玛比率:    {metrics['calmar_ratio']:.2f}")
        print(f"  日均换手:    {metrics['avg_turnover_daily']*100:.1f}%")
        print(f"  胜率:        {metrics['win_rate']*100:.1f}%")
        print(f"  盈利天数:    {metrics['profit_days']}")
        print(f"  亏损天数:    {metrics['loss_days']}")
        print(f"{'='*50}\n")
