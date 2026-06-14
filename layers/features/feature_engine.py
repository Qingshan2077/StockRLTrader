"""
⚠️ 此模块已弃用 — 请使用 layers/features/feature_pipeline.py

原因:
  - FeatureEngine 生成约 30 个因子
  - FeaturePipeline 生成 50+ 个因子, 支持注册表/缓存/特征组管理
  - 两套代码并存会导致特征不一致和重复维护

迁移路径:
  1. 从 FeaturePipeline 导入: from layers.features import FeaturePipeline
  2. 配置特征组: enabled_groups=["trend","momentum","volatility","volume","risk","market"]
  3. run_pipeline.py 已默认使用 FeaturePipeline

本文件保留作为 v1 兼容, 不继续维护。

-------- 原有代码 (只读) --------

特征工程引擎 — 严格时间滚动生成特征，绝对无未来函数泄露

设计原则:
- 所有 rolling/moving 操作均使用历史数据 (在计算前 shift(1))
- 每个特征组独立方法，便于测试和扩展
- 输出列名清晰，方便后续特征选择
"""
import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import Optional


class FeatureEngine:
    """严格时间滚动特征工程，从 OHLCV 生成 30+ 交易特征"""

    PRICE_COLS = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']

    def __init__(self, benchmark_data: Optional[pd.DataFrame] = None):
        """
        Args:
            benchmark_data: 基准数据 (如 SPY)，用于计算 Beta 等市场特征。若为 None 则跳过市场特征。
        """
        self.benchmark_data = benchmark_data
        self.feature_names: list[str] = []
        self._df: Optional[pd.DataFrame] = None

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """输入 raw OHLCV DataFrame，输出带全部特征的 DataFrame"""
        self._df = df.copy()
        self._df.sort_index(inplace=True)

        self._add_returns()
        self._add_volatility()
        self._add_moving_averages()
        self._add_momentum()
        self._add_volume_features()
        self._add_trend_strength()
        self._add_bollinger_bands()
        self._add_drawdown_features()
        self._add_price_range()
        self._add_market_features()

        self._df.dropna(inplace=True)

        self.feature_names = self.get_feature_columns(exclude_price=True)
        return self._df

    def get_feature_columns(self, exclude_price: bool = True) -> list[str]:
        """返回特征列名列表"""
        if self._df is None:
            return []
        if exclude_price:
            return [c for c in self._df.columns if c not in self.PRICE_COLS]
        return list(self._df.columns)

    # ---- 以下每个方法负责一组特征 ----

    def _add_returns(self) -> None:
        """收益率特征: 1/5/10/20 日"""
        close = self._df['Close']
        # 使用 shift(1) 确保 T 日特征不包含 T 日收益
        self._df['ret_1d'] = close.pct_change(1).shift(1)
        self._df['ret_5d'] = close.pct_change(5).shift(1)
        self._df['ret_10d'] = close.pct_change(10).shift(1)
        self._df['ret_20d'] = close.pct_change(20).shift(1)

    def _add_volatility(self) -> None:
        """滚动波动率: 5/10/20 日年化"""
        ret = self._df['Close'].pct_change().shift(1)
        self._df['vol_5d'] = ret.rolling(5).std() * (252 ** 0.5)
        self._df['vol_10d'] = ret.rolling(10).std() * (252 ** 0.5)
        self._df['vol_20d'] = ret.rolling(20).std() * (252 ** 0.5)
        # 波动率变化
        self._df['vol_change'] = self._df['vol_5d'] / self._df['vol_20d'] - 1

    def _add_moving_averages(self) -> None:
        """均线及乖离率"""
        close = self._df['Close']
        self._df['SMA_10'] = ta.sma(close, length=10)
        self._df['SMA_20'] = ta.sma(close, length=20)
        self._df['SMA_50'] = ta.sma(close, length=50)

        # 乖离率 = (价格 - 均线) / 均线 (使用 shift 确保当日价格不泄露)
        shifted = close.shift(1)
        self._df['dist_sma_10'] = (shifted - self._df['SMA_10'].shift(1)) / self._df['SMA_10'].shift(1)
        self._df['dist_sma_20'] = (shifted - self._df['SMA_20'].shift(1)) / self._df['SMA_20'].shift(1)
        self._df['dist_sma_50'] = (shifted - self._df['SMA_50'].shift(1)) / self._df['SMA_50'].shift(1)

        # EMA
        self._df['EMA_12'] = ta.ema(close, length=12)
        self._df['EMA_26'] = ta.ema(close, length=26)

        # 均线交叉信号
        self._df['sma_cross'] = (self._df['SMA_10'] > self._df['SMA_50']).astype(float)

    def _add_momentum(self) -> None:
        """动量指标: RSI, MACD, Stochastic"""
        close = self._df['Close']
        self._df['RSI_14'] = ta.rsi(close, length=14)
        self._df['RSI_28'] = ta.rsi(close, length=28)

        # MACD (12, 26, 9)
        macd = ta.macd(close, fast=12, slow=26, signal=9)
        if macd is not None and not macd.empty:
            macd_cols = macd.columns.tolist()
            self._df['MACD'] = macd[macd_cols[0]]
            self._df['MACD_hist'] = macd[macd_cols[1]]
            self._df['MACD_signal'] = macd[macd_cols[2]]

        # Stochastic
        stoch = ta.stoch(self._df['High'], self._df['Low'], close)
        if stoch is not None and not stoch.empty:
            stoch_cols = stoch.columns.tolist()
            self._df['STOCH_K'] = stoch[stoch_cols[0]]
            if len(stoch_cols) > 1:
                self._df['STOCH_D'] = stoch[stoch_cols[1]]

        # CCI
        self._df['CCI'] = ta.cci(self._df['High'], self._df['Low'], close, length=20)

    def _add_volume_features(self) -> None:
        """成交量特征"""
        vol = self._df['Volume']
        self._df['vol_sma_20'] = ta.sma(vol, length=20)
        self._df['vol_ratio_20'] = vol / self._df['vol_sma_20']
        self._df['vol_change'] = vol.pct_change().shift(1)
        # OBV (能量潮)
        self._df['OBV'] = ta.obv(self._df['Close'], vol)

    def _add_trend_strength(self) -> None:
        """趋势强度"""
        self._df['trend_strength'] = (
            (self._df['SMA_10'] - self._df['SMA_50']) / self._df['SMA_50']
        )

    def _add_bollinger_bands(self) -> None:
        """布林带特征"""
        bb = ta.bbands(self._df['Close'], length=20, std=2)
        if bb is not None and not bb.empty:
            bb_cols = bb.columns.tolist()
            upper = next(c for c in bb_cols if 'U' in c.upper() or 'Upper' in c)
            mid = next(c for c in bb_cols if 'M' in c.upper() or 'Mid' in c)
            lower = next(c for c in bb_cols if 'L' in c.upper() or 'Lower' in c)
            self._df['bb_width'] = (bb[upper] - bb[lower]) / bb[mid]
            self._df['bb_position'] = (self._df['Close'].shift(1) - bb[lower]) / (bb[upper] - bb[lower])

    def _add_drawdown_features(self) -> None:
        """回撤特征"""
        close = self._df['Close']
        self._df['dd_20d'] = (
            close.shift(1).rolling(20).apply(
                lambda x: (x.iloc[-1] / x.max() - 1) if len(x) > 0 else 0
            )
        )

    def _add_price_range(self) -> None:
        """价格形态特征"""
        o, h, l, c = self._df['Open'], self._df['High'], self._df['Low'], self._df['Close']
        self._df['range'] = (h - l) / c.shift(1)
        self._df['body'] = (c - o) / o
        # 上影线 / 下影线比例
        self._df['upper_shadow'] = (h - np.maximum(o, c)) / (h - l + 1e-10)
        self._df['lower_shadow'] = (np.minimum(o, c) - l) / (h - l + 1e-10)

    def _add_market_features(self) -> None:
        """市场敏感度特征 (Beta)"""
        if self.benchmark_data is None:
            self._df['beta_60d'] = 1.0
            return

        # 对齐日期
        common_idx = self._df.index.intersection(self.benchmark_data.index)
        if len(common_idx) < 60:
            self._df['beta_60d'] = 1.0
            return

        stock_ret = self._df['Close'].pct_change().shift(1)
        bench_ret = self.benchmark_data['Close'].pct_change().shift(1)

        def rolling_beta(stock, bench, window=60):
            cov = stock.rolling(window).cov(bench)
            var = bench.rolling(window).var()
            return cov / var.replace(0, np.nan)

        self._df['beta_60d'] = rolling_beta(stock_ret, bench_ret, 60)
        self._df['beta_60d'].fillna(1.0, inplace=True)

    def __repr__(self) -> str:
        n = len(self.feature_names) if self.feature_names else 0
        return f"FeatureEngine(features={n})"
