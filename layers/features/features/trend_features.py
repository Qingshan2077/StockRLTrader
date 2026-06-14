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

        # 所有价格类特征使用 shift(1) 确保 T 日特征只用到 T-1 日及之前的数据
        close_lag = close.shift(1)

        # 简单移动平均
        df['SMA_10'] = ta.sma(close_lag, length=10)
        df['SMA_20'] = ta.sma(close_lag, length=20)
        df['SMA_50'] = ta.sma(close_lag, length=50)
        df['SMA_200'] = ta.sma(close_lag, length=200)

        # 指数移动平均
        df['EMA_12'] = ta.ema(close_lag, length=12)
        df['EMA_26'] = ta.ema(close_lag, length=26)

        # 乖离率
        df['dist_sma_10'] = (close_lag - df['SMA_10']) / df['SMA_10']
        df['dist_sma_50'] = (close_lag - df['SMA_50']) / df['SMA_50']
        df['dist_sma_200'] = (close_lag - df['SMA_200']) / df['SMA_200']

        # MACD
        macd = ta.macd(close_lag, fast=12, slow=26, signal=9)
        if macd is not None and not macd.empty:
            macd_cols = macd.columns.tolist()
            df['MACD'] = macd[macd_cols[0]]
            df['MACD_hist'] = macd[macd_cols[1]]
            df['MACD_signal'] = macd[macd_cols[2]]

        # 趋势强度（基于已 shift 的 SMA，无泄漏）
        df['trend_strength'] = (df['SMA_10'] - df['SMA_50']) / df['SMA_50']

        # 均线交叉
        df['sma_cross'] = (df['SMA_10'] > df['SMA_50']).astype(float)
        df['sma_distance'] = (df['SMA_10'] - df['SMA_200']) / df['SMA_200']

        return df
