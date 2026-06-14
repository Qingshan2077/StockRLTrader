"""
共享工具函数
"""
import numpy as np
import random
import os


def set_seed(seed: int = 42) -> None:
    """固定 numpy / random / torch 随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def compute_drawdown(nav_series: np.ndarray) -> np.ndarray:
    """
    计算回撤序列 (每点相对于历史峰值的百分比下跌)
    """
    running_max = np.maximum.accumulate(nav_series)
    drawdown = (nav_series - running_max) / running_max
    return drawdown


def annualized_sharpe(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """年化夏普比率"""
    if len(returns) < 2:
        return 0.0
    std = np.std(returns)
    if std == 0 or np.isnan(std):
        return 0.0
    return float(np.mean(returns) / std * np.sqrt(periods_per_year))


def annualized_return(nav_series: np.ndarray, n_days: int = None) -> float:
    """从净值序列计算年化收益率"""
    if n_days is None:
        n_days = len(nav_series)
    if n_days < 2:
        return 0.0
    total_return = nav_series[-1] / nav_series[0] - 1
    return float((1 + total_return) ** (252 / n_days) - 1)


def annualized_volatility(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """年化波动率"""
    if len(returns) < 2:
        return 0.0
    return float(np.std(returns) * np.sqrt(periods_per_year))


def sortino_ratio(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """Sortino比率 (下行波动率)"""
    if len(returns) < 2:
        return 0.0
    downside = returns[returns < 0]
    if len(downside) < 2 or np.std(downside) == 0:
        return 0.0
    return float(np.mean(returns) / np.std(downside) * np.sqrt(periods_per_year))


def max_drawdown_value(nav_series: np.ndarray) -> float:
    """最大回撤 (最大负值)"""
    dd = compute_drawdown(nav_series)
    return float(np.min(dd))
