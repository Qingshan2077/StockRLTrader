"""
YAML 配置加载器 — 多文件合并 + 命令行覆盖 + 环境变量注入

加载优先级 (低 → 高):
    1. config/default.yaml
    2. config/{module}.yaml (model/risk/rl/backtest/data)
    3. 命令行 --config 参数
    4. 环境变量 (QUANT_ 前缀)
"""
import yaml
import os
from pathlib import Path
from copy import deepcopy
from typing import Any, Optional


class ConfigLoader:
    """统一配置加载器"""

    def __init__(self, config_dir: str = "config",
                 override_file: Optional[str] = None):
        self.config_dir = Path(config_dir)
        self._config: dict[str, Any] = {}
        self._load_all(override_file)

    # ---------- 加载 ----------

    def _load_all(self, override_file: Optional[str] = None) -> None:
        """加载所有 YAML 文件并合并"""
        # 1. 基础配置
        self._merge_file("default.yaml")

        # 2. 模块配置
        for module in ["model", "risk", "rl", "backtest", "data"]:
            self._merge_file(f"{module}.yaml")

        # 3. 命令行覆盖
        if override_file and Path(override_file).exists():
            self._merge_file(override_file, required=False)

        # 4. 环境变量注入
        self._apply_env_overrides()

    def _merge_file(self, filename: str, required: bool = False) -> None:
        """加载并深度合并 YAML 文件"""
        filepath = self.config_dir / filename
        if not filepath.exists():
            if required:
                raise FileNotFoundError(f"配置文件不存在: {filepath}")
            return

        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        self._config = self._deep_merge(self._config, data)

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> dict:
        """深度合并两个 dict, override 的值覆盖 base"""
        result = deepcopy(base)
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigLoader._deep_merge(result[key], value)
            else:
                result[key] = deepcopy(value)
        return result

    def _apply_env_overrides(self) -> None:
        """应用环境变量覆盖 (QUANT_ 前缀)"""
        for key, value in os.environ.items():
            if not key.startswith("QUANT_"):
                continue
            config_path = self._resolve_env_path(key[6:])
            if config_path:
                self._set_nested(self._config, config_path, self._cast_value(value))

    def _resolve_env_path(self, raw_key: str) -> Optional[str]:
        """
        Convert a QUANT_* environment key into a dotted config path.

        Supported forms:
          QUANT_system__data_dir -> system.data_dir
          QUANT_system_data_dir  -> system.data_dir
          QUANT_data_split_train_ratio -> data_split.train_ratio
        """
        normalized = raw_key.strip().lower()
        if not normalized:
            return None

        aliases = {
            "seed": "system.seed",
            "data_dir": "system.data_dir",
            "output_dir": "system.output_dir",
            "log_dir": "system.log_dir",
            "model_dir": "system.model_dir",
            "data_provider": "market_data.provider",
            "cache_format": "market_data.cache_format",
            "log_level": "system.log_level",
        }
        if normalized in aliases:
            return aliases[normalized]

        if "__" in normalized:
            return ".".join(part for part in normalized.split("__") if part)

        tokens = [part for part in normalized.split("_") if part]
        if not tokens:
            return None

        return self._match_config_path(tokens, self._config) or normalized

    @staticmethod
    def _match_config_path(tokens: list[str], current: Any) -> Optional[str]:
        """Resolve underscore-separated tokens against existing config keys."""
        if not tokens or not isinstance(current, dict):
            return None

        for end in range(len(tokens), 0, -1):
            candidate = "_".join(tokens[:end])
            if candidate not in current:
                continue

            if end == len(tokens):
                return candidate

            suffix = ConfigLoader._match_config_path(tokens[end:], current[candidate])
            if suffix:
                return f"{candidate}.{suffix}"

        return None

    @staticmethod
    def _set_nested(d: dict, path: str, value: Any) -> None:
        """按点号分隔的路径设置嵌套字典值"""
        keys = path.split(".")
        for key in keys[:-1]:
            if key not in d:
                d[key] = {}
            d = d[key]
        d[keys[-1]] = value

    @staticmethod
    def _cast_value(value: str) -> Any:
        """字符串 → Python 类型转换"""
        # bool
        if value.lower() in ("true", "yes", "on"):
            return True
        if value.lower() in ("false", "no", "off"):
            return False
        # int
        try:
            return int(value)
        except ValueError:
            pass
        # float
        try:
            return float(value)
        except ValueError:
            pass
        # list
        if value.startswith("[") and value.endswith("]"):
            import json
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        return value

    # ---------- 访问 ----------

    def get(self, path: str, default: Any = None) -> Any:
        """按点号分隔的路径获取配置值"""
        keys = path.split(".")
        current = self._config
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current

    def to_dict(self) -> dict:
        """返回完整配置字典的深拷贝"""
        return deepcopy(self._config)

    @property
    def config(self) -> dict:
        return self._config

    def __repr__(self) -> str:
        return f"ConfigLoader(files={len(self._config)} top-level keys)"
