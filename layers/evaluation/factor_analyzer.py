"""
因子分析器 — IC/RankIC 追踪，因子收益归因，衰减监控

核心分析维度:
  - 每日因子 IC (截面 Spearman Rank IC)
  - 因子收益 (long-short 多空组合收益)
  - 分位数分层回测 (Q1~Q5)
  - 因子 IC 衰减曲线 (滚动半衰期)
  - 因子相关性热图

输出: 可用于前端看板直接渲染的数据结构
"""

import numpy as np
import pandas as pd
from typing import Optional
from scipy.stats import spearmanr, pearsonr


class FactorAnalyzer:
    """多因子截面分析器"""

    def __init__(self, panel: pd.DataFrame):
        """
        Args:
            panel: MultiIndex DataFrame (date, code) × [factor1, factor2, ..., future_return, sector]
        """
        self.panel = panel
        self.dates = panel.index.get_level_values("date").unique().sort_values()
        self.codes = panel.index.get_level_values("code").unique()
        self._factor_ics: Optional[pd.DataFrame] = None  # date × factor
        self._factor_returns: Optional[pd.DataFrame] = None

    # ── IC 分析 ──

    def compute_factor_ics(
        self,
        factor_cols: list[str],
        return_col: str = "future_return",
    ) -> pd.DataFrame:
        """计算每个因子每日的截面 Rank IC

        Returns:
            DataFrame: date × factor, 值为 Rank IC
        """
        records = []
        for dt in self.dates:
            chunk = self.panel.loc[dt]
            y = chunk[return_col]
            if y.std() < 1e-12:
                continue
            row = {"date": dt}
            for col in factor_cols:
                if col not in chunk.columns:
                    continue
                x = chunk[col].dropna()
                y_ = y[x.index]
                if len(x) < 30:
                    row[col] = np.nan
                    continue
                try:
                    rho, _ = spearmanr(x, y_)
                    row[col] = rho
                except Exception:
                    row[col] = np.nan
            records.append(row)

        df = pd.DataFrame(records).set_index("date")
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        self._factor_ics = df
        return df

    def factor_ic_summary(self) -> pd.DataFrame:
        """因子 IC 汇总统计"""
        if self._factor_ics is None:
            return pd.DataFrame()
        df = self._factor_ics
        summary = pd.DataFrame({
            "IC_mean": df.mean(),
            "IC_std": df.std(ddof=1),
            "IC_IR": df.mean() / df.std(ddof=1).replace(0, np.nan),
            "IC_positive_pct": (df > 0).mean(),
            "t_stat": df.mean() / df.std(ddof=1).replace(0, np.nan) * np.sqrt(len(df)),
        })
        summary["p_value"] = 2 * (1 - _norm_cdf(abs(summary["t_stat"])))
        return summary

    def factor_ic_decay(
        self,
        factor_cols: list[str],
        lookback: int = 252,
    ) -> pd.DataFrame:
        """因子 IC 衰减: 滚动 N 日的 IC 均值曲线

        Returns:
            DataFrame: date × factor, 值为 rolling IC 均值
        """
        if self._factor_ics is None:
            self.compute_factor_ics(factor_cols)
        return self._factor_ics.rolling(lookback, min_periods=60).mean()

    # ── 因子收益分析 ──

    def compute_factor_returns(
        self,
        factor_cols: list[str],
        return_col: str = "future_return",
        top_pct: float = 0.2,
        bottom_pct: float = 0.2,
    ) -> pd.DataFrame:
        """计算每个因子的多空组合收益

        对每个日期:
          取因子值 Top 20% 做多, Bottom 20% 做空
          计算多空收益差

        Returns:
            DataFrame: date × factor_return (多空收益)
        """
        records = []
        for dt in self.dates:
            chunk = self.panel.loc[dt]
            row = {"date": dt}
            n = len(chunk)
            top_n = max(int(n * top_pct), 5)
            bot_n = max(int(n * bottom_pct), 5)
            for col in factor_cols:
                if col not in chunk.columns:
                    continue
                valid = chunk[[col, return_col]].dropna()
                if len(valid) < 50:
                    row[col] = np.nan
                    continue
                sorted_df = valid.sort_values(col, ascending=False)
                top_ret = sorted_df[return_col].iloc[:top_n].mean()
                bot_ret = sorted_df[return_col].iloc[-bot_n:].mean()
                row[col] = top_ret - bot_ret
            records.append(row)

        df = pd.DataFrame(records).set_index("date")
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        self._factor_returns = df
        return df

    def factor_return_cumulative(
        self,
    ) -> pd.DataFrame:
        """因子多空收益累计曲线"""
        if self._factor_returns is None:
            return pd.DataFrame()
        return (1 + self._factor_returns).cumprod()

    # ── 分位数分层 ──

    def quantile_analysis(
        self,
        factor_col: str,
        return_col: str = "future_return",
        n_buckets: int = 5,
    ) -> pd.DataFrame:
        """因子分位数分层回测

        Returns:
            DataFrame: bucket × mean_return, sharpe, t_stat
        """
        # 每日分桶
        rets = []
        for dt in self.dates:
            chunk = self.panel.loc[dt]
            valid = chunk[[factor_col, return_col]].dropna()
            if len(valid) < n_buckets * 3:
                continue
            # 将因子值映射到 0..n_buckets-1 整数桶
            scale = (n_buckets - 1) / max(valid[factor_col].max() - valid[factor_col].min(), 1e-10)
            valid["bucket_idx"] = (valid[factor_col] - valid[factor_col].min()) * scale
            valid["bucket_idx"] = valid["bucket_idx"].clip(0, n_buckets - 1).astype(int)
            for b in range(n_buckets):
                mask_b = valid["bucket_idx"] == b
                if mask_b.sum() < 2:
                    continue
                rets.append({
                    "date": dt,
                    "bucket": f"Q{b+1}",
                    "return": valid.loc[mask_b, return_col].mean(),
                })

        df = pd.DataFrame(rets)
        if df.empty:
            return pd.DataFrame({"mean_return": [0]*5, "sharpe": [0]*5, "count": [0]*5},
                                 index=[f"Q{i+1}" for i in range(5)])
        summary = df.groupby("bucket")["return"].agg(
            ["mean", "std", "count"]
        ).rename(columns={"mean": "mean_return", "std": "std_return"})
        summary["sharpe"] = summary["mean_return"] / summary["std_return"].replace(0, np.nan) * np.sqrt(252)
        summary["t_stat"] = summary["mean_return"] / summary["std_return"].replace(0, np.nan) * np.sqrt(summary["count"])
        return summary

    # ── 因子相关性 ──

    def factor_correlation(self, factor_cols: list[str]) -> pd.DataFrame:
        """计算因子间平均截面相关性 (跨日期平均)

        Returns:
            DataFrame: factor × factor (Pearson r)
        """
        corr_stack = []
        for dt in self.dates:
            chunk = self.panel.loc[dt]
            avail = [c for c in factor_cols if c in chunk.columns]
            if len(avail) < 2:
                continue
            corr_stack.append(chunk[avail].corr(method="pearson").values)

        mean_corr = np.mean(corr_stack, axis=0)
        return pd.DataFrame(mean_corr, index=avail, columns=avail)

    # ── 最佳因子排名 ──

    def top_factors(
        self,
        factor_cols: list[str],
        return_col: str = "future_return",
        top_n: int = 10,
    ) -> pd.DataFrame:
        """返回按 IC_IR 排序的最佳因子"""
        if self._factor_ics is None:
            self.compute_factor_ics(factor_cols, return_col)
        summary = self.factor_ic_summary()
        summary = summary.sort_values("IC_IR", ascending=False)
        return summary.head(top_n)


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    """标准正态 CDF"""
    from scipy.stats import norm
    return norm.cdf(x)
