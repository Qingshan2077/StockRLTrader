"""
特征注册表 — 管理所有特征的元数据

每个特征:
    - name: 列名
    - group: 特征组 (trend/momentum/volatility/volume/risk/market)
    - params: 参数字典
    - version: 版本号
    - dependencies: 依赖的原始列
"""
from dataclasses import dataclass, field
from typing import Callable, Optional
import hashlib


@dataclass
class FeatureMeta:
    """单个特征的元数据"""
    name: str
    group: str                    # trend / momentum / volatility / volume / risk / market
    description: str = ""
    params: dict = field(default_factory=dict)
    version: str = "1.0"
    dependencies: list[str] = field(default_factory=list)


class FeatureRegistry:
    """特征注册表"""

    def __init__(self):
        self._features: list[FeatureMeta] = []
        self._by_group: dict[str, list[FeatureMeta]] = {}

    def register(self, meta: FeatureMeta) -> None:
        """注册一个特征"""
        if meta.name in {f.name for f in self._features}:
            raise ValueError(f"特征 '{meta.name}' 已存在")
        self._features.append(meta)
        self._by_group.setdefault(meta.group, []).append(meta)

    def register_batch(self, metas: list[FeatureMeta]) -> None:
        """批量注册特征"""
        for meta in metas:
            self.register(meta)

    def get(self, name: str) -> Optional[FeatureMeta]:
        """按名称查找特征"""
        for f in self._features:
            if f.name == name:
                return f
        return None

    def get_group(self, group: str) -> list[FeatureMeta]:
        """获取一个特征组"""
        return self._by_group.get(group, [])

    def list_groups(self) -> list[str]:
        """列出所有特征组"""
        return list(self._by_group.keys())

    def list_names(self) -> list[str]:
        """列出所有特征名"""
        return [f.name for f in self._features]

    def compute_hash(self) -> str:
        """计算特征配置的哈希, 用于缓存键"""
        content = "|".join(
            f"{f.name}:{f.group}:{f.version}:{sorted(f.params.items())}"
            for f in sorted(self._features, key=lambda x: x.name)
        )
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def __len__(self) -> int:
        return len(self._features)

    def __repr__(self) -> str:
        return f"FeatureRegistry({len(self)} features, {len(self._by_group)} groups)"
