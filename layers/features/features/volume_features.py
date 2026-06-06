"""
成交量特征: volume_ratio / OBV / turnover_change / VWAP
"""
import pandas as pd
import pandas_ta as ta


class VolumeFeatures:
    """成交量类特征生成器"""

    def add(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        close, vol = df['Close'], df['Volume']

        # 成交量均线
        df['vol_sma_10'] = ta.sma(vol, length=10)
        df['vol_sma_20'] = ta.sma(vol, length=20)

        # 成交量比率
        df['vol_ratio_10'] = vol / df['vol_sma_10']
        df['vol_ratio_20'] = vol / df['vol_sma_20']

        # 成交量变化
        df['volume_change'] = vol.pct_change().shift(1)

        # OBV
        df['OBV'] = ta.obv(close, vol)

        # VWAP 近似
        df['VWAP_approx'] = (close * vol).cumsum() / vol.cumsum()

        # 换手率变化 (成交量变化率的绝对值移动平均)
        df['turnover_change'] = vol.pct_change().abs().rolling(20).mean()

        return df
