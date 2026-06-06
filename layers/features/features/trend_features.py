"""
趋势特征: MA / EMA / MA_distance / MACD / trend_strength
"""
import pandas as pd
import pandas_ta as ta
import numpy as np


class TrendFeatures:
    """趋势类特征生成器"""

    def add(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        close = df['Close']

        # 简单移动平均
        df['SMA_10'] = ta.sma(close, length=10)
        df['SMA_20'] = ta.sma(close, length=20)
        df['SMA_50'] = ta.sma(close, length=50)
        df['SMA_200'] = ta.sma(close, length=200)

        # 指数移动平均
        df['EMA_12'] = ta.ema(close, length=12)
        df['EMA_26'] = ta.ema(close, length=26)

        # 乖离率 (使用 shift 防泄露)
        shifted = close.shift(1)
        df['dist_sma_10'] = (shifted - df['SMA_10'].shift(1)) / df['SMA_10'].shift(1)
        df['dist_sma_50'] = (shifted - df['SMA_50'].shift(1)) / df['SMA_50'].shift(1)
        df['dist_sma_200'] = (shifted - df['SMA_200'].shift(1)) / df['SMA_200'].shift(1)

        # MACD
        macd = ta.macd(close, fast=12, slow=26, signal=9)
        if macd is not None and not macd.empty:
            macd_cols = macd.columns.tolist()
            df['MACD'] = macd[macd_cols[0]]
            df['MACD_hist'] = macd[macd_cols[1]]
            df['MACD_signal'] = macd[macd_cols[2]]

        # 趋势强度
        df['trend_strength'] = (df['SMA_10'] - df['SMA_50']) / df['SMA_50']

        # 均线交叉
        df['sma_cross'] = (df['SMA_10'] > df['SMA_50']).astype(float)
        df['sma_distance'] = (df['SMA_10'] - df['SMA_200']) / df['SMA_200']

        return df
