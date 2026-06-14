"""
Portfolio Optimizer — 组合优化

支持 (单资产阶段为简化版, 多资产阶段为完整版):
    - Mean-Variance Optimization
    - Risk Parity
    - Minimum Variance
"""
import numpy as np
from scipy.optimize import minimize


class PortfolioOptimizer:
    """组合优化器"""

    def __init__(self, method: str = "mean_variance",
                 risk_aversion: float = 1.0,
                 max_weight: float = 1.0,
                 max_leverage: float = 1.0):
        self.method = method
        self.risk_aversion = risk_aversion
        self.max_weight = max_weight
        self.max_leverage = max_leverage

    def optimize(self, expected_returns: np.ndarray,
                 cov_matrix: np.ndarray,
                 current_weights: np.ndarray = None) -> np.ndarray:
        """
        Args:
            expected_returns: 期望收益 (单资产时是 signal_score)
            cov_matrix: 协方差矩阵 (单资产时用方差标量)
            current_weights: 当前权重

        Returns:
            最优权重向量
        """
        n = len(expected_returns)

        if n == 1:
            # 单资产: 直接按信号调整
            return np.clip(expected_returns, -self.max_weight, self.max_weight)

        if self.method == "mean_variance":
            return self._mean_variance(expected_returns, cov_matrix, n)
        elif self.method == "risk_parity":
            return self._risk_parity(cov_matrix, n)
        elif self.method == "min_variance":
            return self._minimum_variance(cov_matrix, n)
        else:
            raise ValueError(f"未知优化方法: {self.method}")

    def _mean_variance(self, mu: np.ndarray, cov: np.ndarray, n: int) -> np.ndarray:
        """Mean-Variance Optimization"""
        def objective(w):
            return -mu.dot(w) + self.risk_aversion * w.dot(cov).dot(w)

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(np.abs(w)) - 1.0},
        ]
        bounds = [(-self.max_weight, self.max_weight)] * n
        w0 = np.ones(n) / n

        result = minimize(objective, w0, method="SLSQP",
                          bounds=bounds, constraints=constraints,
                          options={"maxiter": 1000, "ftol": 1e-10})
        return result.x if result.success else w0

    def _risk_parity(self, cov: np.ndarray, n: int) -> np.ndarray:
        """Risk Parity (equal risk contribution)"""
        def risk_contribution(w):
            portfolio_vol = np.sqrt(w.dot(cov).dot(w))
            marginal_risk = cov.dot(w)
            rc = w * marginal_risk / portfolio_vol
            return rc

        def objective(w):
            rc = risk_contribution(w)
            target = 1.0 / n
            return np.sum((rc - target) ** 2)

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        ]
        bounds = [(0, self.max_weight)] * n  # long only
        w0 = np.ones(n) / n

        result = minimize(objective, w0, method="SLSQP",
                          bounds=bounds, constraints=constraints)
        return result.x if result.success else w0

    def _minimum_variance(self, cov: np.ndarray, n: int) -> np.ndarray:
        """Minimum Variance Portfolio"""
        def objective(w):
            return w.dot(cov).dot(w)

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        ]
        bounds = [(0, self.max_weight)] * n
        w0 = np.ones(n) / n

        result = minimize(objective, w0, method="SLSQP",
                          bounds=bounds, constraints=constraints)
        return result.x if result.success else w0
