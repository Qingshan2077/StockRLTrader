"""
RL 执行优化环境 — Gymnasium 接口

核心思想: RL 不接触原始 OHLCV，只接收层1(signal_score)和层2(target_position)的输出。
RL 的任务: 在每个时间步输出 execution_ratio ∈ [0,1]，控制调仓节奏。

executed_position = current_position + execution_ratio * (target_position - current_position)
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque


class ExecutionEnv(gym.Env):
    """
    RL 执行优化环境

    状态空间 (约 40 维):
        - signal_score, target_position, current_position, unrealized_pnl
        - recent_volatility_5d/20d, recent_drawdown, estimated_cost
        - recent_rebalance_count, market_regime_features
        - 滑动窗口历史: position, signal, target, execution (各5天)

    动作空间: Box(low=0, high=1) — execution_ratio

    奖励: portfolio_return - λ1*cost - λ2*turnover - λ3*drawdown_penalty
    """

    def __init__(self, signal_scores: np.ndarray,
                 target_positions: np.ndarray,
                 prices: np.ndarray,
                 volumes: np.ndarray,
                 volatilities: np.ndarray,
                 config,
                 window_size: int = 5):
        super().__init__()

        self.signal_scores = signal_scores
        self.target_positions = target_positions
        self.prices = prices
        self.volumes = volumes
        self.volatilities = volatilities
        self.config = config
        self.window_size = window_size

        n_steps = len(signal_scores)
        if not (len(target_positions) == len(prices) == len(volumes) == len(volatilities) == n_steps):
            raise ValueError("所有输入数组长度必须一致")

        self.n_steps = n_steps

        # 状态维度: 核心9 + regime5 + history_4*5(=20) + 辅助6 = 40
        n_core = 9
        n_regime = 5
        n_history = 4 * window_size  # position, signal, target, execution
        n_aux = 6
        self.obs_dim = n_core + n_regime + n_history + n_aux

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        # 历史缓冲区
        self.pos_history = deque(maxlen=window_size)
        self.signal_history = deque(maxlen=window_size)
        self.target_history = deque(maxlen=window_size)
        self.exec_history = deque(maxlen=window_size)

    # ---------- 环境生命周期 ----------

    @property
    def _max_episode_steps(self):
        return self.n_steps - 2

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.config.initial_balance
        self.shares_held = 0.0
        self.portfolio_value = self.config.initial_balance
        self.nav_history = [self.config.initial_balance]
        self.trade_count = 0
        self._entry_price = 0.0
        self._cost_total = 0.0

        for _ in range(self.window_size):
            self.pos_history.append(0.0)
            self.signal_history.append(0.0)
            self.target_history.append(0.0)
            self.exec_history.append(0.0)

        return self._get_observation(), {}

    def step(self, action):
        execution_ratio = float(np.clip(action[0], 0.0, 1.0))

        # 获取当前值
        signal = float(self.signal_scores[self.current_step])
        target = float(self.target_positions[self.current_step])
        price = float(self.prices[self.current_step])
        volume = float(self.volumes[self.current_step])
        volatility = float(self.volatilities[self.current_step])

        current_position = self.shares_held * price / max(self.portfolio_value, 1.0)

        # 计算执行后的目标仓位
        raw_target_delta = target - current_position
        executed_delta = execution_ratio * raw_target_delta
        executed_target = current_position + executed_delta

        # 记录执行前的净值
        prev_net_worth = self.portfolio_value

        # 执行交易
        cost = self._execute_trade(executed_target, price, volume)

        # 更新净值 (使用下一步的价格, 但当前步先按当前价算)
        new_shares_value = self.shares_held * price
        self.portfolio_value = self.balance + new_shares_value
        self.nav_history.append(self.portfolio_value)

        # 更新持仓位置 (比值)
        new_position = self.shares_held * price / max(self.portfolio_value, 1.0)

        # 记录
        self.pos_history.append(current_position)
        self.signal_history.append(signal)
        self.target_history.append(target)
        self.exec_history.append(execution_ratio)

        # 计算奖励
        portfolio_return = (self.portfolio_value - prev_net_worth) / max(prev_net_worth, 1.0)
        reward = self._compute_reward(portfolio_return, cost, executed_delta, raw_target_delta)

        self._cost_total += cost
        self.current_step += 1

        done = self.current_step >= self._max_episode_steps
        truncated = False

        return self._get_observation(), reward, done, truncated, {"cost": cost}

    # ---------- 交易执行 ----------

    def _execute_trade(self, target_position: float, price: float, volume: float) -> float:
        """执行调仓，返回交易成本"""
        current_shares_value = self.shares_held * price
        current_pos_ratio = current_shares_value / max(self.portfolio_value, 1.0)

        target_value = target_position * self.portfolio_value
        delta_value = target_value - current_shares_value

        if abs(delta_value) < 1e-4 * self.portfolio_value:
            return 0.0

        if delta_value > 0:  # 买入
            available_cash = self.balance * 0.95  # 保留 5% 现金缓冲
            buy_value = min(delta_value, available_cash)
            if buy_value <= 0:
                return 0.0
            shares_to_buy = buy_value / price
            cost = buy_value * (self.config.commission + self.config.slippage)
            self.shares_held += shares_to_buy
            self.balance -= buy_value + cost
            self.trade_count += 1
            return float(cost)
        else:  # 卖出
            sell_value = min(-delta_value, current_shares_value)
            if sell_value <= 0:
                return 0.0
            shares_to_sell = sell_value / price
            cost = sell_value * (self.config.commission + self.config.slippage)
            self.shares_held -= shares_to_sell
            self.balance += sell_value - cost
            self.trade_count += 1
            return float(cost)

    # ---------- 奖励函数 ----------

    def _compute_reward(self, portfolio_return: float, cost: float,
                         executed_delta: float, raw_delta: float) -> float:
        """奖励 = 组合收益 - λ1*cost - λ2*turnover - λ3*drawdown_penalty"""
        nav = np.array(self.nav_history)
        drawdown = (nav - np.maximum.accumulate(nav)) / np.maximum.accumulate(nav)
        current_dd = float(drawdown[-1])
        dd_penalty = max(0, abs(current_dd) - self.config.drawdown_threshold)

        reward = (portfolio_return
                  - self.config.lambda_cost * cost
                  - self.config.lambda_turnover * abs(executed_delta)
                  - self.config.lambda_drawdown * dd_penalty)

        return float(np.clip(reward, -10.0, 10.0))

    # ---------- 状态构造 ----------

    def _get_observation(self) -> np.ndarray:
        """构造约 40 维观测向量"""
        step = min(self.current_step, self.n_steps - 1)
        signal = float(self.signal_scores[step])
        target = float(self.target_positions[step])
        price = float(self.prices[step])
        volume = float(self.volumes[step])
        volatility = float(self.volatilities[step])

        current_position = self.shares_held * price / max(self.portfolio_value, 1.0)

        # 入场的平均成本
        if self.shares_held > 1e-6:
            avg_cost = (self.config.initial_balance - self.balance) / self.shares_held
            unrealized_pnl = (price - avg_cost) / max(avg_cost, 1e-6)
        else:
            unrealized_pnl = 0.0

        nav = np.array(self.nav_history)
        running_max = np.maximum.accumulate(nav)
        drawdown = (nav[-1] - running_max[-1]) / max(running_max[-1], 1.0)

        estimated_cost = (self.config.commission + self.config.slippage) * abs(target - current_position)

        # 核心 9 维
        core = np.array([
            signal, target, current_position, unrealized_pnl,
            volatility, drawdown, estimated_cost,
            float(min(self.trade_count, 10)) / 10.0,
            self.balance / max(self.config.initial_balance, 1.0),
        ], dtype=np.float32)

        # 市场 regime 5 维
        regime = np.array([
            np.tanh(signal * 3),
            np.clip(volatility / 0.3 - 1, -1, 1),
            np.tanh(np.log1p(volume) / 10 - 1),
            np.clip(drawdown * 20, -1, 1),
            1.0 if target > 0 else -1.0 if target < 0 else 0.0,
        ], dtype=np.float32)

        # 滑动窗口历史 (各 5 天)
        history = np.concatenate([
            list(self.pos_history),
            list(self.signal_history),
            list(self.target_history),
            list(self.exec_history),
        ]).astype(np.float32)

        # 辅助 6 维
        aux = np.array([
            float(step) / max(self.n_steps, 1),
            self.shares_held / max((self.config.initial_balance / max(price, 1e-6)), 1.0),
            nav[-1] / max(self.config.initial_balance, 1.0)
            if len(nav) > 1 else 1.0,
            (nav[-1] - nav[-2]) / max(nav[-2], 1e-6) if len(nav) > 1 else 0.0,
            float(self.trade_count),
            self._cost_total / max(self.config.initial_balance, 1.0),
        ], dtype=np.float32)

        obs = np.concatenate([core, regime, history, aux])
        return np.nan_to_num(obs, nan=0.0).astype(np.float32)
