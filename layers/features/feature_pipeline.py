from __future__ import annotations

"""
特征管线 v2 — 可配置的特征生成流程

支持:
    - rolling window / expanding window
    - 特征组选择性启用
    - 输出 feature metadata
    - 统一接入 FeatureCache 缓存
"""

from typing import Optional

import pandas as pd

from layers.features.feature_cache import FeatureCache
from layers.features.feature_registry import FeatureMeta, FeatureRegistry


class FeaturePipeline:
    """可配置的特征生成管线。

    负责按分组生成趋势、动量、波动率、成交量、风险和市场类因子，并统一做
    shift(1) 处理，避免当前时点的特征直接使用当前收盘价带来的未来函数风险。
    """

    def __init__(
        self,
        registry: FeatureRegistry | None = None,
        cache: FeatureCache | None = None,
        benchmark_data: Optional[pd.DataFrame] = None,
        enabled_groups: list[str] | None = None,
    ):
        self.registry = registry or FeatureRegistry()
        self.cache = cache or FeatureCache()
        self.benchmark_data = benchmark_data
        self.enabled_groups = enabled_groups or [
            "trend",
            "momentum",
            "volatility",
            "volume",
            "risk",
            "market",
        ]

    def build_features(self, df: pd.DataFrame, ticker: str = "unknown", use_cache: bool = True) -> pd.DataFrame:
        """执行完整特征生成流程。

        Args:
            df: 原始 OHLCV 行情数据。
            ticker: 股票代码，用于特征缓存文件命名。
            use_cache: 是否读取和写入本地特征缓存。

        Returns:
            包含原始价格列和全部生成因子的 DataFrame。
        """
        if df is None or df.empty:
            return df

        df = self._normalize_columns(df)

        if use_cache:
            cached = self.cache.get(ticker, self.registry)
            if cached is not None and len(cached) > 10:
                return cached

        result = df.copy()
        result.sort_index(inplace=True)

        from layers.features.features import (
            MarketFeatures,
            MomentumFeatures,
            RiskFeatures,
            TrendFeatures,
            VolatilityFeatures,
            VolumeFeatures,
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
            if group_name not in self.enabled_groups:
                continue
            old_cols = set(result.columns)
            result = engine.add(result)
            for col in set(result.columns) - old_cols:
                self.registry.register(FeatureMeta(name=col, group=group_name))

        price_cols = {"Open", "High", "Low", "Close", "Volume", "Adj Close"}
        feature_cols = [col for col in result.columns if col not in price_cols]
        # 所有非价格原始列统一滞后一日，确保因子在时间 t 不直接使用 Close[t]。
        result[feature_cols] = result[feature_cols].shift(1)
        result.dropna(inplace=True)

        if use_cache:
            self.cache.save(result, ticker, self.registry)

        return result

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化 OHLCV 列名，兼容 open/high/low/close/volume 小写输入。"""
        rename = {
            col: col.capitalize()
            for col in df.columns
            if col.lower() in {"open", "high", "low", "close", "volume"}
        }
        return df.rename(columns=rename)

    @property
    def feature_count(self) -> int:
        return len(self.registry)

    def __repr__(self) -> str:
        return f"FeaturePipeline(groups={self.enabled_groups}, features={self.feature_count})"
