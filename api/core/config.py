from functools import lru_cache
from typing import Any

from utils.config_loader import ConfigLoader


@lru_cache(maxsize=1)
def get_config_loader() -> ConfigLoader:
    return ConfigLoader()


def get_config(path: str, default: Any = None) -> Any:
    return get_config_loader().get(path, default)


def get_config_dict() -> dict:
    return get_config_loader().to_dict()
