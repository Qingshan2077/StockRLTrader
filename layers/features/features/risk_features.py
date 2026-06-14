"""
风险特征: beta / rolling_drawdown / correlation / downside_ratio
"""
import pandas as pd
import numpy as np


class RiskFeatures:
    """风险类特征生成器"""

    def __init__(self, benchmark_data: pd.DataFrame = None):
        self.benchmark_data = benchmark_data

    def add(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        close = df['Close']
        ret = close.pct_change().shift(1)

        # 滚动最大回撤
        df['dd_20d'] = close.shift(1).rolling(20).apply(
            lambda x: (x.iloc[-1] / x.max() - 1) if len(x) > 0 and x.max() > 0 else 0
        )
        df['dd_60d'] = close.shift(1).rolling(60).apply(
            lambda x: (x.iloc[-1] / x.max() - 1) if len(x) > 0 and x.max() > 0 else 0
        )

        # 回撤深度
        df['dd_depth'] = df['dd_20d'].rolling(20).min()

        # 下行风险指标
        df['downside_ratio'] = (
            ret.clip(upper=0).rolling(20).std() /
            ret.rolling(20).std().replace(0, np.nan)
        )

        # Beta (如果有基准数据)
        if self.benchmark_data is not None:
            bench_ret = self.benchmark_data['Close'].pct_change().shift(1)
            common_idx = df.index.intersection(bench_ret.index)
            if len(common_idx) > 60:
                stock_slice = ret.loc[common_idx]
                bench_slice = bench_ret.loc[common_idx]
                cov = stock_slice.rolling(60).cov(bench_slice)
                var = bench_slice.rolling(60).var()
                df['beta_60d'] = cov / var.replace(0, np.nan)
            else:
                df['beta_60d'] = 1.0
        else:
            df['beta_60d'] = 1.0

        df['beta_60d'] = df['beta_60d'].fillna(1.0)
        return df
