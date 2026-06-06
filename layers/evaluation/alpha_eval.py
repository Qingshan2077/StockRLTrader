"""
Alpha Evaluation — IC / RankIC / IC decay / SHAP / 因子分析
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from typing import Optional


class AlphaEvaluator:
    """Alpha 信号质量评估器"""

    def __init__(self, signal_scores: np.ndarray,
                 forward_returns: np.ndarray,
                 dates: Optional[pd.DatetimeIndex] = None):
        """
        Args:
            signal_scores: alpha 信号值 (1D array)
            forward_returns: 对应的未来收益率 (1D array, 已对齐)
            dates: 日期索引 (可选, 用于 IC 时间序列)
        """
        if len(signal_scores) != len(forward_returns):
            raise ValueError("signal_scores 和 forward_returns 长度必须一致")
        mask = ~(np.isnan(signal_scores) | np.isnan(forward_returns))
        self.signal = signal_scores[mask]
        self.fwd_ret = forward_returns[mask]
        self.dates = dates[mask] if dates is not None else None
        self.n = len(self.signal)

    # ---------- IC 分析 ----------

    def compute_ic(self) -> dict:
        """计算 IC (Pearson correlation) 及其统计显著性"""
        if self.n < 3:
            return {"IC": 0.0, "IC_pvalue": 1.0, "IC_tstat": 0.0}
        ic, pval = pearsonr(self.signal, self.fwd_ret)
        # t-stat = r * sqrt((n-2) / (1-r^2))
        tstat = float(ic * np.sqrt((self.n - 2) / (1 - ic**2 + 1e-16)))
        return {
            "IC": float(ic),
            "IC_pvalue": float(pval),
            "IC_tstat": tstat,
        }

    def compute_rank_ic(self) -> dict:
        """计算 Rank IC (Spearman rank correlation)"""
        if self.n < 3:
            return {"RankIC": 0.0, "RankIC_tstat": 0.0}
        ric, pval = spearmanr(self.signal, self.fwd_ret)
        return {
            "RankIC": float(ric),
            "RankIC_pvalue": float(pval),
        }

    def compute_ic_series(self, window: int = 21) -> dict:
        """计算滚动 IC 时间序列 (无需 dates 也能工作)"""
        if self.n < window:
            return {}

        ic_series = []
        for i in range(window, self.n):
            s = self.signal[i - window:i]
            r = self.fwd_ret[i - window:i]
            if np.std(s) == 0 or np.std(r) == 0:
                continue
            ic, _ = pearsonr(s, r)
            ic_series.append(ic)

        if not ic_series:
            return {}

        ic_arr = np.array(ic_series)
        ic_mean = float(np.mean(ic_arr))
        ic_std = float(np.std(ic_arr))
        n_eff = len(ic_arr)
        return {
            "IC_mean": ic_mean,
            "IC_std": ic_std,
            "IC_tstat": float(ic_mean / (ic_std / np.sqrt(n_eff)) if ic_std > 0 else 0),
            "IC_IR": float(ic_mean / ic_std if ic_std > 0 else 0),
            "IC_series": ic_arr.tolist(),
        }

    def compute_ic_decay(self, horizons: list[int] = None,
                          all_fwd_returns: dict = None) -> dict:
        """计算 IC decay (不同 horizon 的 IC 衰减)"""
        if horizons is None:
            horizons = [1, 3, 5, 10, 20]
        if all_fwd_returns is None:
            return {}

        decay = {}
        for h in horizons:
            key = f'fwd_return_{h}d' if f'fwd_return_{h}d' in all_fwd_returns else f'label_ret_{h}d'
            if key not in all_fwd_returns:
                continue
            ret = all_fwd_returns[key].values if hasattr(all_fwd_returns[key], 'values') else all_fwd_returns[key]
            mask = ~(np.isnan(self.signal) | np.isnan(ret[:len(self.signal)]))
            if mask.sum() < 3:
                continue
            ic, _ = pearsonr(self.signal[mask], ret[:len(self.signal)][mask])
            decay[f"IC_{h}d"] = float(ic)
        return decay

    # ---------- 信号分析 ----------

    def compute_turnover(self) -> float:
        """信号换手率: 信号值的日均变化率"""
        if self.n < 2:
            return 0.0
        changes = np.abs(np.diff(self.signal))
        return float(np.mean(changes))

    def compute_quantile_analysis(self, n_buckets: int = 5) -> dict:
        """分位数分析: 信号分数从低到高, 每组平均收益"""
        if self.n < n_buckets:
            return {}

        # 按信号大小分桶
        quantiles = np.percentile(self.signal, np.linspace(0, 100, n_buckets + 1))
        result = {}
        for i in range(n_buckets):
            mask = (self.signal >= quantiles[i]) & (self.signal < quantiles[i + 1])
            if i == n_buckets - 1:
                mask = self.signal >= quantiles[i]  # 最后一组包含边界
            bucket_ret = np.mean(self.fwd_ret[mask]) if mask.sum() > 0 else 0.0
            result[f"Q{i+1}_return"] = float(bucket_ret)
            result[f"Q{i+1}_count"] = int(mask.sum())

        # Top-Bottom spread
        q1_mask = self.signal >= quantiles[-2]
        q5_mask = self.signal < quantiles[1]
        top = np.mean(self.fwd_ret[q1_mask]) if q1_mask.sum() > 0 else 0
        bottom = np.mean(self.fwd_ret[q5_mask]) if q5_mask.sum() > 0 else 0
        result["top_bottom_spread"] = float(top - bottom)

        return result

    def compute_factor_correlation(self, other_signals: dict) -> dict:
        """因子相关性矩阵 (与其他信号/特征的相关性)"""
        corr = {}
        for name, other in other_signals.items():
            if len(other) != self.n:
                continue
            mask = ~(np.isnan(other))
            if mask.sum() < 3:
                continue
            c, _ = pearsonr(self.signal[mask], other[mask])
            corr[name] = float(c)
        return corr

    def full_report(self) -> dict:
        """生成完整评估报告"""
        report = {}
        report.update(self.compute_ic())
        report.update(self.compute_rank_ic())
        report.update(self.compute_ic_series())
        report["signal_turnover"] = self.compute_turnover()
        report["n_samples"] = self.n

        buckets = self.compute_quantile_analysis()
        if buckets:
            report["quantile_analysis"] = buckets

        return report

    def __repr__(self) -> str:
        ic = self.compute_ic().get("IC", 0)
        return f"AlphaEvaluator(n={self.n}, IC={ic:.4f})"
