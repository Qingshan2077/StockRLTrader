from __future__ import annotations

"""
特征注册表 — 管理所有特征的元数据

每个特征记录:
    - name: 列名
    - group: 特征组 (trend/momentum/volatility/volume/risk/market)
    - params: 参数字典
    - version: 版本号
    - dependencies: 依赖的原始列
"""

from dataclasses import dataclass, field
import hashlib
from typing import Optional


@dataclass
class FeatureMeta:
    """单个因子的元数据。"""

    name: str
    group: str
    description: str = ""
    params: dict = field(default_factory=dict)
    version: str = "1.0"
    dependencies: list[str] = field(default_factory=list)


class FeatureRegistry:
    """因子注册表，用于记录因子名称、分组和版本信息。"""

    def __init__(self):
        self._features: list[FeatureMeta] = []
        self._by_group: dict[str, list[FeatureMeta]] = {}

    def register(self, meta: FeatureMeta) -> None:
        """注册一个因子，重复名称会直接报错，避免后续训练时列名歧义。"""
        if meta.name in {feature.name for feature in self._features}:
            raise ValueError(f"Feature '{meta.name}' already exists")
        self._features.append(meta)
        self._by_group.setdefault(meta.group, []).append(meta)

    def register_batch(self, metas: list[FeatureMeta]) -> None:
        """批量注册因子。"""
        for meta in metas:
            self.register(meta)

    def get(self, name: str) -> Optional[FeatureMeta]:
        """按名称查找因子元数据。"""
        for feature in self._features:
            if feature.name == name:
                return feature
        return None

    def get_group(self, group: str) -> list[FeatureMeta]:
        """获取某个因子分组下的全部因子。"""
        return self._by_group.get(group, [])

    def list_groups(self) -> list[str]:
        """列出当前已注册的所有因子分组。"""
        return list(self._by_group.keys())

    def list_names(self) -> list[str]:
        """列出当前已注册的所有因子名称。"""
        return [feature.name for feature in self._features]

    def compute_hash(self) -> str:
        """计算因子配置哈希，用于特征缓存键。"""
        content = "|".join(
            f"{feature.name}:{feature.group}:{feature.version}:{sorted(feature.params.items())}"
            for feature in sorted(self._features, key=lambda item: item.name)
        )
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def __len__(self) -> int:
        return len(self._features)

    def __repr__(self) -> str:
        return f"FeatureRegistry({len(self)} features, {len(self._by_group)} groups)"
