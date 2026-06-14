"""
市场特征: regime / relative_strength / spread
"""
import pandas as pd
import numpy as np


class MarketFeatures:
    """市场环境特征生成器"""

    def __init__(self, benchmark_data: pd.DataFrame = None):
        self.benchmark_data = benchmark_data

    def add(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        close = df['Close']
        ret = close.pct_change().shift(1)

        # ---- 简单 Regime 分类 (基于 trend + vol + volume) ----
        # Trend regime: +1 (上升), 0 (震荡), -1 (下降)
        sma_20 = close.shift(1).rolling(20).mean()
        sma_50 = close.shift(1).rolling(50).mean()
        trend_regime = np.where(sma_20 > sma_50 * 1.02, 1,
                                np.where(sma_20 < sma_50 * 0.98, -1, 0))
        df['regime_trend'] = trend_regime

        # Vol regime: +1 (高波动), 0 (正常), -1 (低波动)
        vol = ret.rolling(20).std()
        vol_median = vol.rolling(252).median()
        vol_ratio = vol / vol_median.replace(0, np.nan)
        df['regime_vol'] = np.where(vol_ratio > 1.5, 1,
                                    np.where(vol_ratio < 0.5, -1, 0))

        # Volume regime
        vol_chg = df['Volume'].pct_change().rolling(20).mean()
        df['regime_volume'] = np.where(vol_chg > 0.2, 1,
                                        np.where(vol_chg < -0.2, -1, 0))

        # Composite regime [-3, 3]
        df['regime_composite'] = (
            df['regime_trend'] + df['regime_vol'] + df['regime_volume']
        )

        # ---- 相对强弱 (vs 自身历史) ----
        df['rel_strength_60'] = close.shift(1) / close.shift(1).rolling(60).mean() - 1
        df['rel_strength_252'] = close.shift(1) / close.shift(1).rolling(252).mean() - 1

        # ---- Benchmark 相对强度 ----
        if self.benchmark_data is not None:
            bench_close = self.benchmark_data['Close']
            common_idx = df.index.intersection(bench_close.index)
            if len(common_idx) > 60:
                stock_norm = close.loc[common_idx] / close.loc[common_idx].iloc[0]
                bench_norm = bench_close.loc[common_idx] / bench_close.loc[common_idx].iloc[0]
                df['vs_benchmark'] = stock_norm.values / bench_norm.values - 1
            else:
                df['vs_benchmark'] = 0.0
        else:
            df['vs_benchmark'] = 0.0

        return df
