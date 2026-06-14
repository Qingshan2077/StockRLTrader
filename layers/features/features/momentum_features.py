"""
动量特征: RSI / Stochastic / Momentum / CCI
"""
import pandas as pd
import pandas_ta as ta


class MomentumFeatures:
    """动量类特征生成器"""

    def add(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        close, high, low = df['Close'], df['High'], df['Low']

        # 所有价格类特征使用 shift(1) 确保 T 日特征只用到 T-1 日及之前的数据
        close_lag = close.shift(1)
        high_lag = high.shift(1)
        low_lag = low.shift(1)

        # RSI
        df['RSI_14'] = ta.rsi(close_lag, length=14)
        df['RSI_28'] = ta.rsi(close_lag, length=28)

        # Stochastic
        stoch = ta.stoch(high_lag, low_lag, close_lag)
        if stoch is not None and not stoch.empty:
            sc = stoch.columns.tolist()
            df['STOCH_K'] = stoch[sc[0]]
            if len(sc) > 1:
                df['STOCH_D'] = stoch[sc[1]]

        # Momentum
        df['momentum_10'] = ta.mom(close_lag, length=10)
        df['momentum_20'] = ta.mom(close_lag, length=20)

        # CCI
        df['CCI_20'] = ta.cci(high_lag, low_lag, close_lag, length=20)

        # 价格变化率
        df['roc_5'] = close_lag.pct_change(5)
        df['roc_10'] = close_lag.pct_change(10)

        return df
