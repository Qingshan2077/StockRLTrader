"""
波动率特征: rolling_std / ATR / realized_vol / BB / downside_vol
"""
import pandas as pd
import pandas_ta as ta
import numpy as np


class VolatilityFeatures:
    """波动率类特征生成器"""

    def add(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        close, high, low = df['Close'], df['High'], df['Low']

        # 收益率序列 (shift 防泄露)
        ret = close.pct_change().shift(1)

        # 滚动年化波动率
        for w in [5, 10, 20]:
            df[f'vol_{w}d'] = ret.rolling(w).std() * (252 ** 0.5)

        # ATR
        df['ATR_14'] = ta.atr(high, low, close, length=14)

        # 已实现波动率 (年化)
        df['realized_vol'] = ret.rolling(20).std() * (252 ** 0.5)

        # 下行波动率
        df['downside_vol'] = ret.clip(upper=0).rolling(20).std() * (252 ** 0.5)

        # 波动率变化
        df['vol_change'] = df['vol_5d'] / df['vol_20d'] - 1

        # 布林带
        bb = ta.bbands(close, length=20, std=2)
        if bb is not None and not bb.empty:
            bb_cols = bb.columns.tolist()
            upper = next(c for c in bb_cols if 'U' in c.upper())
            mid = next(c for c in bb_cols if 'M' in c.upper())
            lower = next(c for c in bb_cols if 'L' in c.upper())
            df['bb_width'] = (bb[upper] - bb[lower]) / bb[mid]
            df['bb_position'] = (close.shift(1) - bb[lower]) / (bb[upper] - bb[lower])

        return df
