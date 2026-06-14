"""
特征管线 v2 — 可配置的特征生成流程

支持:
    - rolling window / expanding window
    - 特征组选择性启用
    - 输出 feature metadata
"""
import pandas as pd
from typing import Optional
from layers.features.feature_registry import FeatureRegistry, FeatureMeta
from layers.features.feature_cache import FeatureCache


class FeaturePipeline:
    """可配置的特征管线"""

    def __init__(self, registry: FeatureRegistry = None,
                 cache: FeatureCache = None,
                 benchmark_data: Optional[pd.DataFrame] = None,
                 enabled_groups: list[str] = None):
        self.registry = registry or FeatureRegistry()
        self.cache = cache or FeatureCache()
        self.benchmark_data = benchmark_data

        # 默认启用所有组
        self.enabled_groups = enabled_groups or [
            "trend", "momentum", "volatility", "volume", "risk", "market"
        ]

    def build_features(self, df: pd.DataFrame, ticker: str = "unknown",
                       use_cache: bool = True) -> pd.DataFrame:
        """
        执行特征生成管线

        Args:
            df: 原始 OHLCV DataFrame
            ticker: 股票代码 (用于缓存)
            use_cache: 是否使用缓存

        Returns:
            包含所有特征的 DataFrame
        """
        if df is None or df.empty:
            return df

        # 标准化列名 (确保首字母大写: open→Open)
        df = self._normalize_columns(df)

        # 尝试缓存
        if use_cache:
            cached = self.cache.get(ticker, self.registry)
            if cached is not None and len(cached) > 10:
                return cached

        result = df.copy()
        result.sort_index(inplace=True)

        # 分特征组生成
        from layers.features.features import (
            TrendFeatures, MomentumFeatures, VolatilityFeatures,
            VolumeFeatures, RiskFeatures, MarketFeatures
        )

        group_map = {
            "trend": TrendFeatures(),
            "momentum": MomentumFeatures(),
            "volatility": VolatilityFeatures(),
            "volume": VolumeFeatures(),
            "risk": RiskFeatures(self.benchmark_data),
            "market": MarketFeatures(self.benchmark_data),
        }

        for group_name, engine in group_map.items():
            if group_name in self.enabled_groups:
                old_cols = set(result.columns)
                result = engine.add(result)
                new_cols = set(result.columns) - old_cols
                for col in new_cols:
                    self.registry.register(FeatureMeta(name=col, group=group_name))

        # ⚠️ 防前瞻泄漏：所有非OHLCV特征列统一 shift(1)
        # 确保特征在时间t不使用Close[t]（与标签分母共享的变量）
        _price_cols = {'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'}
        _feat_cols = [c for c in result.columns if c not in _price_cols]
        result[_feat_cols] = result[_feat_cols].shift(1)

        result.dropna(inplace=True)

        # 写缓存
        if use_cache:
            self.cache.save(result, ticker, self.registry)

        return result

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化列名为首字母大写"""
        rename = {c: c.capitalize() for c in df.columns
                  if c.lower() in ('open', 'high', 'low', 'close', 'volume')}
        return df.rename(columns=rename)

    @property
    def feature_count(self) -> int:
        return len(self.registry)

    def __repr__(self) -> str:
        return (f"FeaturePipeline(groups={self.enabled_groups}, "
                f"features={self.feature_count})")
