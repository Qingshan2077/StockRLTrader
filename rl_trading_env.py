import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class TradingEnvironment(gym.Env):
    """
    股票交易环境 - 用于强化学习训练

    状态空间：技术指标 + 持仓信息
    动作空间：买入、卖出、持有
    奖励函数：基于收益和风险的综合评分
    """

    def __init__(self, df, initial_balance=10000, commission=0.001, window_size=10):
        super(TradingEnvironment, self).__init__()

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.commission = commission  # 交易手续费率
        self.window_size = window_size  # 观察窗口大小

        # 提取特征列（排除价格列）
        self.feature_columns = [col for col in df.columns if col not in
                                ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]

        # 动作空间：0=持有, 1=买入, 2=卖出
        self.action_space = spaces.Discrete(3)

        # 状态空间：技术指标 + 持仓信息
        # 特征数量 * 窗口大小 + 额外状态（余额、持仓、当前价格等）
        n_features = len(self.feature_columns) * window_size + 6
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32
        )

        # 初始化环境状态
        self.reset()

    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)

        # 从一个随机位置开始（但要保证有足够的历史数据）
        self.current_step = self.window_size

        # 交易状态
        self.balance = self.initial_balance  # 可用现金
        self.shares_held = 0  # 持有股数
        self.total_shares_bought = 0
        self.total_shares_sold = 0

        # 交易历史
        self.trades = []
        self.net_worths = [self.initial_balance]

        return self._get_observation(), {}

    def _get_observation(self):
        """获取当前状态观察"""
        # 1. 获取技术指标的历史窗口
        start = max(0, self.current_step - self.window_size)
        end = self.current_step

        # 提取特征并归一化
        features = []
        for col in self.feature_columns:
            window_data = self.df[col].iloc[start:end].values

            # 处理 NaN 和 inf
            window_data = np.nan_to_num(window_data, nan=0.0, posinf=1.0, neginf=-1.0)

            # 稳健的归一化（使用 tanh）
            if len(window_data) > 0:
                # 使用中位数和 MAD 进行稳健归一化
                median = np.median(window_data)
                mad = np.median(np.abs(window_data - median))

                if mad > 1e-8:  # 避免除以零
                    normalized = (window_data - median) / (mad * 1.4826)  # MAD to STD conversion
                    # 使用 tanh 限制在 [-1, 1] 范围
                    normalized = np.tanh(normalized / 3.0)
                else:
                    normalized = np.zeros_like(window_data)
            else:
                normalized = np.zeros(0)

            # 如果窗口不够，用0填充
            if len(normalized) < self.window_size:
                normalized = np.pad(normalized, (self.window_size - len(normalized), 0), constant_values=0)

            features.extend(normalized)

        # 2. 添加当前持仓信息
        current_price = self.df.loc[self.current_step, 'Close']
        shares_value = self.shares_held * current_price
        total_value = self.balance + shares_value

        # 归一化并限制范围
        position_info = [
            np.clip(self.balance / self.initial_balance, 0, 10),
            np.clip(self.shares_held / 100, 0, 10),
            np.clip(shares_value / self.initial_balance, 0, 10),
            np.clip(total_value / self.initial_balance, 0, 10),
            np.clip(current_price / 100, 0, 100),
            np.clip((total_value - self.initial_balance) / self.initial_balance, -1, 5)
        ]

        # 确保没有 NaN 或 inf
        position_info = [np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0) for x in position_info]

        features.extend(position_info)

        obs = np.array(features, dtype=np.float32)

        # 最后的安全检查
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        obs = np.clip(obs, -10, 10)  # 限制在合理范围内

        return obs

    def step(self, action):
        """执行动作"""
        current_price = self.df.loc[self.current_step, 'Close']
        prev_net_worth = self.balance + self.shares_held * current_price

        # 执行交易动作
        if action == 1:  # 买入
            self._buy(current_price)
        elif action == 2:  # 卖出
            self._sell(current_price)
        # action == 0 则持有，不做任何操作

        # 移动到下一步
        self.current_step += 1

        # 计算新的净值
        if self.current_step < len(self.df):
            new_price = self.df.loc[self.current_step, 'Close']
            new_net_worth = self.balance + self.shares_held * new_price
        else:
            new_net_worth = prev_net_worth

        self.net_worths.append(new_net_worth)

        # 计算奖励
        reward = self._calculate_reward(prev_net_worth, new_net_worth, action)

        # 判断是否结束
        done = self.current_step >= len(self.df) - 1
        truncated = False

        # 如果破产，提前结束
        if new_net_worth <= 0:
            done = True
            reward = -100  # 破产惩罚

        observation = self._get_observation() if not done else np.zeros(self.observation_space.shape)

        return observation, reward, done, truncated, {}

    def _buy(self, price):
        """买入股票"""
        # 使用80%的余额买入（保留一些现金）
        max_shares = int(self.balance * 0.8 / price)

        if max_shares > 0:
            cost = max_shares * price * (1 + self.commission)

            if cost <= self.balance:
                self.balance -= cost
                self.shares_held += max_shares
                self.total_shares_bought += max_shares

                self.trades.append({
                    'step': self.current_step,
                    'type': 'buy',
                    'price': price,
                    'shares': max_shares,
                    'cost': cost
                })

    def _sell(self, price):
        """卖出股票"""
        if self.shares_held > 0:
            revenue = self.shares_held * price * (1 - self.commission)
            self.balance += revenue

            self.trades.append({
                'step': self.current_step,
                'type': 'sell',
                'price': price,
                'shares': self.shares_held,
                'revenue': revenue
            })

            self.total_shares_sold += self.shares_held
            self.shares_held = 0

    def _calculate_reward(self, prev_net_worth, new_net_worth, action):
        """
        计算奖励函数
        考虑因素：
        1. 净值变化（收益）
        2. 风险（波动率）
        3. 交易频率（减少过度交易）
        """
        # 避免除以零
        if prev_net_worth <= 0:
            prev_net_worth = 1.0

        # 基础奖励：净值变化百分比
        net_worth_change = (new_net_worth - prev_net_worth) / prev_net_worth
        # 限制奖励范围，避免极端值
        net_worth_change = np.clip(net_worth_change, -0.1, 0.1)
        reward = net_worth_change * 100

        # 惩罚过度交易
        if action != 0:  # 如果进行了交易
            reward -= 0.1

        # 奖励趋势把握
        if len(self.net_worths) >= 5:
            recent_trend = np.diff(self.net_worths[-5:]).mean()
            if recent_trend > 0 and self.shares_held > 0:
                reward += 0.5
            elif recent_trend < 0 and self.shares_held == 0:
                reward += 0.5

        # 风险调整（夏普比率思想）
        if len(self.net_worths) >= 11:

            # 计算最近10个收益率
            recent_values = np.array(self.net_worths[-11:])  # 取最近11个值
            returns = np.diff(recent_values) / recent_values[:-1]  # 计算10个收益率
            returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

            if len(returns) > 0 and returns.std() > 1e-8:
                sharpe = returns.mean() / returns.std()
                sharpe = np.clip(sharpe, -5, 5)
                reward += sharpe * 0.1

        # 确保奖励没有 NaN
        reward = np.nan_to_num(reward, nan=0.0, posinf=1.0, neginf=-1.0)
        reward = np.clip(reward, -10, 10)

        return float(reward)

    def get_portfolio_value(self):
        """获取当前投资组合价值"""
        if self.current_step < len(self.df):
            current_price = self.df.loc[self.current_step, 'Close']
            return self.balance + self.shares_held * current_price
        return self.balance

    def get_metrics(self):
        """获取交易指标"""
        final_value = self.get_portfolio_value()
        total_return = (final_value - self.initial_balance) / self.initial_balance

        # 计算最大回撤
        net_worths = np.array(self.net_worths)
        running_max = np.maximum.accumulate(net_worths)
        drawdown = (net_worths - running_max) / running_max
        max_drawdown = drawdown.min()

        # 交易次数
        num_trades = len(self.trades)

        # 收益率序列
        if len(self.net_worths) > 1:
            returns = np.diff(self.net_worths) / self.net_worths[:-1]
            sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0
        else:
            sharpe_ratio = 0

        return {
            'total_return': total_return,
            'final_value': final_value,
            'max_drawdown': max_drawdown,
            'num_trades': num_trades,
            'sharpe_ratio': sharpe_ratio,
            'total_shares_bought': self.total_shares_bought,
            'total_shares_sold': self.total_shares_sold
        }

    def render(self):
        """渲染环境（可选）"""
        if self.current_step < len(self.df):
            current_price = self.df.loc[self.current_step, 'Close']
            portfolio_value = self.get_portfolio_value()

            print(f"Step: {self.current_step}")
            print(f"Price: ${current_price:.2f}")
            print(f"Balance: ${self.balance:.2f}")
            print(f"Shares: {self.shares_held}")
            print(f"Portfolio Value: ${portfolio_value:.2f}")
            print(f"Total Return: {((portfolio_value - self.initial_balance) / self.initial_balance * 100):.2f}%")
            print("-" * 50)