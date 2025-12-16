import numpy as np
import pandas as pd
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import torch as th
import pickle
from pathlib import Path
from rl_trading_env import TradingEnvironment


class TrainingCallback(BaseCallback):
    """训练过程回调，用于记录训练进度和实时指标"""

    def __init__(self, verbose=0, update_freq=100):
        super(TrainingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0
        self.update_freq = update_freq

        # 训练指标
        self.timesteps = []
        self.losses = []
        self.learning_rates = []
        self.value_losses = []
        self.policy_losses = []
        self.entropy_losses = []

        # 交易指标
        self.portfolio_values = []
        self.actions_history = []
        self.rewards_history = []
        self.balance_history = []
        self.shares_history = []

        # 当前 episode 的统计
        self.current_episode_reward = 0
        self.current_episode_length = 0

    def _on_training_start(self):
        """训练开始时调用"""
        self.episode_count = 0

    def _on_step(self):
        """每步调用"""
        # 记录当前步的信息
        if len(self.locals.get('rewards', [])) > 0:
            reward = self.locals['rewards'][0]
            self.current_episode_reward += reward
            self.rewards_history.append(reward)

        self.current_episode_length += 1

        # 记录环境信息（如果可用）
        if 'infos' in self.locals and len(self.locals['infos']) > 0:
            info = self.locals['infos'][0]

            # 从环境获取交易信息
            if hasattr(self.training_env.envs[0], 'balance'):
                self.balance_history.append(self.training_env.envs[0].balance)
                self.shares_history.append(self.training_env.envs[0].shares_held)
                portfolio_value = self.training_env.envs[0].get_portfolio_value()
                self.portfolio_values.append(portfolio_value)

        # 记录动作
        if 'actions' in self.locals:
            action = int(self.locals['actions'][0])
            self.actions_history.append(action)

        # Episode 结束
        if self.locals.get('dones', [False])[0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.episode_count += 1

            # 重置当前 episode 统计
            self.current_episode_reward = 0
            self.current_episode_length = 0

        # 每隔一定步数记录训练指标
        if self.n_calls % self.update_freq == 0:
            self.timesteps.append(self.n_calls)

            # 尝试获取训练指标
            if hasattr(self.model, 'logger') and self.model.logger is not None:
                # 获取 loss 等指标
                try:
                    # PPO 特有指标
                    if hasattr(self.model, 'policy_loss'):
                        self.policy_losses.append(float(self.model.policy_loss))
                    if hasattr(self.model, 'value_loss'):
                        self.value_losses.append(float(self.model.value_loss))

                    # 学习率
                    if hasattr(self.model, 'learning_rate'):
                        lr = self.model.learning_rate
                        if callable(lr):
                            lr = lr(1)  # 获取当前学习率
                        self.learning_rates.append(float(lr))
                except:
                    pass

        return True

    def get_statistics(self):
        """获取训练统计信息"""
        stats = {
            'total_timesteps': self.n_calls,
            'episode_count': self.episode_count,
            'avg_episode_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
            'avg_episode_length': np.mean(self.episode_lengths[-100:]) if self.episode_lengths else 0,
            'current_portfolio_value': self.portfolio_values[-1] if self.portfolio_values else 0,
            'current_balance': self.balance_history[-1] if self.balance_history else 0,
            'current_shares': self.shares_history[-1] if self.shares_history else 0,
            'total_rewards': sum(self.rewards_history),
            'buy_count': self.actions_history.count(1),
            'sell_count': self.actions_history.count(2),
            'hold_count': self.actions_history.count(0),
        }
        return stats


class RLTradingAgent:
    """
    强化学习交易代理
    使用 PPO 算法学习交易策略
    """

    def __init__(self, data, model_type='PPO', model_path=None):
        """
        初始化 RL Agent

        Args:
            data: DataFrame，包含技术指标的股票数据
            model_type: 'PPO' 或 'A2C'
            model_path: 预训练模型路径（可选）
        """
        self.data = data
        self.model_type = model_type
        self.model = None
        self.env = None
        self.training_history = {
            'returns': [],
            'sharpe_ratios': [],
            'max_drawdowns': [],
            'num_trades': []
        }

        if model_path and Path(model_path).exists():
            self.load_model(model_path)

    def create_environment(self, initial_balance=10000, commission=0.001):
        """创建交易环境"""
        env = TradingEnvironment(
            df=self.data,
            initial_balance=initial_balance,
            commission=commission,
            window_size=10
        )
        self.env = DummyVecEnv([lambda: env])
        return self.env

    def train(self, total_timesteps=100000, initial_balance=10000,
              learning_rate=0.0003, verbose=1, progress_callback=None):
        """
        训练 RL Agent

        Args:
            total_timesteps: 总训练步数
            initial_balance: 初始资金
            learning_rate: 学习率
            verbose: 是否打印训练信息
            progress_callback: 进度回调函数，用于实时更新界面
        """
        print(f"\n{'=' * 60}")
        print(f"开始训练 {self.model_type} Agent")
        print(f"{'=' * 60}")
        print(f"训练步数: {total_timesteps}")
        print(f"初始资金: ${initial_balance}")
        print(f"学习率: {learning_rate}")
        print(f"数据长度: {len(self.data)} 天\n")

        # 创建环境
        if self.env is None:
            self.create_environment(initial_balance=initial_balance)

        # 创建回调（支持进度回调）
        callback = TrainingCallback(verbose=verbose, update_freq=50)

        # 如果提供了进度回调函数，添加自定义回调
        if progress_callback:
            callback.progress_callback = progress_callback

        # 创建模型（使用更稳定的参数）
        if self.model_type == 'PPO':
            self.model = PPO(
                'MlpPolicy',
                self.env,
                learning_rate=learning_rate,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                clip_range_vf=None,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                policy_kwargs=dict(
                    net_arch=[dict(pi=[256, 256], vf=[256, 256])],
                    activation_fn=th.nn.Tanh
                ),
                verbose=verbose
            )
        elif self.model_type == 'A2C':
            self.model = A2C(
                'MlpPolicy',
                self.env,
                learning_rate=learning_rate,
                gamma=0.99,
                gae_lambda=0.95,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                policy_kwargs=dict(
                    net_arch=[dict(pi=[256, 256], vf=[256, 256])],
                    activation_fn=th.nn.Tanh
                ),
                verbose=verbose
            )
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")

        # 训练模型
        print("开始训练...\n")
        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callback,
                progress_bar=False  # 禁用默认进度条，使用自定义
            )

            # 保存训练统计
            self.training_history['callback_stats'] = callback.get_statistics()
            self.training_history['episode_rewards'] = callback.episode_rewards
            self.training_history['portfolio_values'] = callback.portfolio_values
            self.training_history['actions_history'] = callback.actions_history

        except Exception as e:
            print(f"\n训练过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            raise

        print("\n训练完成！")

        # 评估训练结果
        metrics = self.evaluate()

        return metrics

    def evaluate(self, num_episodes=1):
        """评估模型性能"""
        if self.model is None:
            raise ValueError("模型尚未训练或加载")

        all_metrics = []

        for episode in range(num_episodes):
            # 重置环境
            obs = self.env.reset()
            done = False

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)

            # 获取该episode的指标
            metrics = self.env.envs[0].get_metrics()
            all_metrics.append(metrics)

            # 记录到历史
            self.training_history['returns'].append(metrics['total_return'])
            self.training_history['sharpe_ratios'].append(metrics['sharpe_ratio'])
            self.training_history['max_drawdowns'].append(metrics['max_drawdown'])
            self.training_history['num_trades'].append(metrics['num_trades'])

        # 平均指标
        avg_metrics = {
            'total_return': np.mean([m['total_return'] for m in all_metrics]),
            'final_value': np.mean([m['final_value'] for m in all_metrics]),
            'max_drawdown': np.mean([m['max_drawdown'] for m in all_metrics]),
            'num_trades': np.mean([m['num_trades'] for m in all_metrics]),
            'sharpe_ratio': np.mean([m['sharpe_ratio'] for m in all_metrics])
        }

        print(f"\n{'=' * 60}")
        print("评估结果")
        print(f"{'=' * 60}")
        print(f"总收益率: {avg_metrics['total_return'] * 100:.2f}%")
        print(f"最终资产: ${avg_metrics['final_value']:.2f}")
        print(f"最大回撤: {avg_metrics['max_drawdown'] * 100:.2f}%")
        print(f"夏普比率: {avg_metrics['sharpe_ratio']:.3f}")
        print(f"交易次数: {avg_metrics['num_trades']:.0f}")
        print(f"{'=' * 60}\n")

        return avg_metrics

    def predict_action(self, observation):
        """预测下一步动作"""
        if self.model is None:
            raise ValueError("模型尚未训练或加载")

        action, _ = self.model.predict(observation, deterministic=True)

        # 转换为可读的动作
        action_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
        return action_map[int(action)]

    def get_trading_signals(self, recent_data, initial_balance=10000):
        """
        获取交易信号

        Args:
            recent_data: 最近的数据（包含技术指标）
            initial_balance: 初始资金

        Returns:
            dict: 包含动作和建议的字典
        """
        if self.model is None:
            raise ValueError("模型尚未训练或加载")

        # 创建临时环境
        temp_env = TradingEnvironment(
            df=recent_data,
            initial_balance=initial_balance,
            window_size=10
        )

        obs, _ = temp_env.reset()
        action, _ = self.model.predict(obs, deterministic=True)

        action_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
        action_str = action_map[int(action)]

        # 获取当前状态信息
        current_price = recent_data.iloc[-1]['Close']

        # 生成建议
        suggestions = {
            'action': action_str,
            'price': current_price,
            'confidence': 'HIGH',  # 可以根据模型输出的概率来判断
            'reasoning': self._generate_reasoning(action_str, recent_data)
        }

        return suggestions

    def _generate_reasoning(self, action, data):
        """生成交易理由"""
        latest = data.iloc[-1]

        reasons = []

        # 基于技术指标的理由
        if 'RSI' in data.columns:
            rsi = latest['RSI']
            if rsi > 70:
                reasons.append(f"RSI={rsi:.1f} (超买)")
            elif rsi < 30:
                reasons.append(f"RSI={rsi:.1f} (超卖)")

        if 'MACD_12_26_9' in data.columns and 'MACDs_12_26_9' in data.columns:
            macd = latest['MACD_12_26_9']
            signal = latest['MACDs_12_26_9']
            if macd > signal:
                reasons.append("MACD金叉")
            else:
                reasons.append("MACD死叉")

        # 趋势判断
        if 'SMA_10' in data.columns and 'SMA_50' in data.columns:
            sma10 = latest['SMA_10']
            sma50 = latest['SMA_50']
            if sma10 > sma50:
                reasons.append("短期均线上穿长期均线(上涨趋势)")
            else:
                reasons.append("短期均线下穿长期均线(下跌趋势)")

        # 价格动量
        if 'Returns' in data.columns:
            returns = latest['Returns']
            if returns > 0:
                reasons.append(f"价格上涨 {returns * 100:.2f}%")
            else:
                reasons.append(f"价格下跌 {abs(returns) * 100:.2f}%")

        if not reasons:
            reasons.append("基于历史模式识别")

        return "; ".join(reasons)

    def save_model(self, path):
        """保存模型"""
        if self.model is None:
            raise ValueError("没有可保存的模型")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # 保存模型
        self.model.save(str(path))

        # 保存训练历史
        history_path = path.parent / f"{path.stem}_history.pkl"
        with open(history_path, 'wb') as f:
            pickle.dump(self.training_history, f)

        print(f"✅ 模型已保存到: {path}")

    def load_model(self, path):
        """加载模型"""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"模型文件不存在: {path}")

        # 加载模型
        if self.model_type == 'PPO':
            self.model = PPO.load(str(path))
        elif self.model_type == 'A2C':
            self.model = A2C.load(str(path))

        # 加载训练历史
        history_path = path.parent / f"{path.stem}_history.pkl"
        if history_path.exists():
            with open(history_path, 'rb') as f:
                self.training_history = pickle.load(f)

        print(f"✅ 模型已加载: {path}")

    def backtest(self, test_data, initial_balance=10000):
        """
        回测模型

        Args:
            test_data: 测试数据
            initial_balance: 初始资金

        Returns:
            dict: 回测结果和交易记录
        """
        if self.model is None:
            raise ValueError("模型尚未训练或加载")

        # 创建测试环境
        test_env = TradingEnvironment(
            df=test_data,
            initial_balance=initial_balance,
            window_size=10
        )

        obs, _ = test_env.reset()
        done = False

        actions_taken = []
        prices = []

        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = test_env.step(action)

            actions_taken.append(int(action))
            if test_env.current_step < len(test_data):
                prices.append(test_data.iloc[test_env.current_step]['Close'])

        # 获取回测指标
        metrics = test_env.get_metrics()
        trades = test_env.trades
        net_worths = test_env.net_worths

        results = {
            'metrics': metrics,
            'trades': trades,
            'net_worths': net_worths,
            'actions': actions_taken,
            'prices': prices
        }

        print(f"\n{'=' * 60}")
        print("回测结果")
        print(f"{'=' * 60}")
        print(f"总收益率: {metrics['total_return'] * 100:.2f}%")
        print(f"最终资产: ${metrics['final_value']:.2f}")
        print(f"最大回撤: {metrics['max_drawdown'] * 100:.2f}%")
        print(f"夏普比率: {metrics['sharpe_ratio']:.3f}")
        print(f"交易次数: {metrics['num_trades']}")
        print(f"{'=' * 60}\n")

        return results


# 测试代码
if __name__ == "__main__":
    from improved_data_engine import DataEngine

    # 加载数据
    print("加载数据...")
    engine = DataEngine("AAPL")
    df = engine.load_processed_data()

    if df is None:
        print("请先下载数据")
        exit(1)

    # 分割训练集和测试集
    split = int(len(df) * 0.8)
    train_data = df.iloc[:split]
    test_data = df.iloc[split:]

    print(f"训练集: {len(train_data)} 天")
    print(f"测试集: {len(test_data)} 天")

    # 创建并训练 Agent
    agent = RLTradingAgent(train_data, model_type='PPO')

    # 训练
    agent.train(total_timesteps=50000, initial_balance=10000)

    # 回测
    results = agent.backtest(test_data, initial_balance=10000)

    # 保存模型
    agent.save_model("data/models/rl_agent_AAPL.zip")