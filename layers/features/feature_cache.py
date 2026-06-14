"""
因子缓存 — 基于配置哈希的特征缓存

缓存键 = md5(ticker + features_hash + date_range)
缓存位置: stock_data/{ticker}_features_{hash}.parquet
"""
import hashlib
import pandas as pd
from pathlib import Path
from typing import Optional
from layers.features.feature_registry import FeatureRegistry


class FeatureCache:
    """因子缓存管理器"""

    def __init__(self, data_dir: str = "stock_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._hits = 0
        self._misses = 0

    def _cache_key(self, ticker: str, registry: FeatureRegistry,
                   start_date: str = None, end_date: str = None,
                   data_hash: str = None) -> str:
        """生成缓存键 (ticker + features_hash + date_range + data_hash)"""
        parts = [
            ticker.upper(),
            registry.compute_hash(),
            start_date or "all",
            end_date or "all",
            data_hash or "no_data_hash",
        ]
        return hashlib.md5("|".join(parts).encode()).hexdigest()[:16]

    def _data_content_hash(self, df) -> str:
        """计算数据内容哈希 (取样加速)"""
        sample = df.head(3).to_csv() + df.tail(3).to_csv() + str(len(df))
        return hashlib.md5(sample.encode()).hexdigest()[:8]

    def _cache_path(self, ticker: str, cache_key: str) -> Path:
        return self.data_dir / f"{ticker}_features_{cache_key}.parquet"

    def get(self, ticker: str, registry: FeatureRegistry,
            start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
        """尝试从缓存加载特征（仅按 registry hash + date 匹配，不检查 data hash）"""
        key = self._cache_key(ticker, registry, start_date, end_date)
        path = self._cache_path(ticker, key)

        if path.exists():
            self._hits += 1
            return pd.read_parquet(path)

        # 兼容动态注册因子的场景：FeaturePipeline 在生成特征后才知道完整 registry，
        # 因此首次读取时 registry 可能还是空的。精确键未命中时，回退到该股票最新缓存。
        candidates = sorted(
            self.data_dir.glob(f"{ticker.upper()}_features_*.parquet"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            self._hits += 1
            return pd.read_parquet(candidates[0])

        self._misses += 1
        return None

    def save(self, features: pd.DataFrame, ticker: str, registry: FeatureRegistry,
             start_date: str = None, end_date: str = None) -> None:
        """缓存特征数据（加入原始数据哈希）"""
        data_hash = self._data_content_hash(features)
        key = self._cache_key(ticker, registry, start_date, end_date, data_hash)
        path = self._cache_path(ticker, key)
        features.to_parquet(path)
        latest_key = self._cache_key(ticker, registry, start_date, end_date)
        latest_path = self._cache_path(ticker, latest_key)
        if latest_path != path:
            features.to_parquet(latest_path)

    def clear(self, ticker: str = None) -> None:
        """清除缓存"""
        pattern = f"{ticker}_features_*.parquet" if ticker else "*_features_*.parquet"
        for f in self.data_dir.glob(pattern):
            f.unlink()

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def __repr__(self) -> str:
        return f"FeatureCache(hits={self._hits}, misses={self._misses}, rate={self.hit_rate:.1%})"
