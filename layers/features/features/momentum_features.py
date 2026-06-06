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

        # RSI
        df['RSI_14'] = ta.rsi(close, length=14)
        df['RSI_28'] = ta.rsi(close, length=28)

        # Stochastic
        stoch = ta.stoch(high, low, close)
        if stoch is not None and not stoch.empty:
            sc = stoch.columns.tolist()
            df['STOCH_K'] = stoch[sc[0]]
            if len(sc) > 1:
                df['STOCH_D'] = stoch[sc[1]]

        # Momentum
        df['momentum_10'] = ta.mom(close, length=10)
        df['momentum_20'] = ta.mom(close, length=20)

        # CCI
        df['CCI_20'] = ta.cci(high, low, close, length=20)

        # 价格变化率
        df['roc_5'] = close.pct_change(5).shift(1)
        df['roc_10'] = close.pct_change(10).shift(1)

        return df
