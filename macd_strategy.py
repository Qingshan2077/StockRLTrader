import numpy as np
import pandas as pd
from datetime import datetime


class MACDStrategy:
    """
    基于MACD指标的短线交易策略

    买入信号：
    - MACD快线由下跌变为平滑（变化 <= 0.02）
    - 绿柱缩短（柱状图为负且在增加）

    卖出信号：
    - MACD快线由上升变为平滑（变化 <= 0.02）
    """

    def __init__(self, data, initial_balance=10000, commission=0.001, smooth_threshold=0.02):
        """
        初始化策略

        Args:
            data: DataFrame，包含MACD指标的股票数据
            initial_balance: 初始资金
            commission: 手续费率（默认0.1%）
            smooth_threshold: 平滑阈值（默认0.02）
        """
        self.data = data.copy()
        self.initial_balance = initial_balance
        self.commission = commission
        self.smooth_threshold = smooth_threshold

        # 确保有MACD指标
        required_cols = ['MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9']
        # MACD_12_26_9	MACD 指标的主线（也常被叫做 DIF）
        # MACDs_12_26_9	MACD 的信号线（Signal / DEA）
        # MACDh_12_26_9	MACD 的柱状图（Histogram）
        for col in required_cols:
            if col not in self.data.columns:
                raise ValueError(f"数据中缺少 {col} 列，请先计算MACD指标")

        # 交易记录
        self.trades = []
        self.positions = []  # 持仓记录
        self.balance_history = []
        self.portfolio_values = []

        # 当前状态
        self.balance = initial_balance
        self.shares = 0
        self.position_open = False
        self.buy_price = 0

    def generate_signals(self):
        """
        生成买卖信号

        Returns:
            DataFrame: 包含信号的数据
        """
        df = self.data.copy()

        # 计算MACD变化
        df['MACD_Change'] = df['MACD_12_26_9'].diff()
        df['MACD_Change_Prev'] = df['MACD_Change'].shift(1)

        # 计算柱状图变化
        df['MACDh_Change'] = df['MACDh_12_26_9'].diff()

        # 初始化信号
        df['Signal'] = 0  # 0=持有, 1=买入, -1=卖出
        df['Signal_Reason'] = ''  # 记录信号原因

        for i in range(2, len(df)):
            # ============ 买入信号 ============
            # 条件1: 前一天MACD下跌
            macd_was_falling = df['MACD_Change'].iloc[i - 1] < 0

            # 条件2: 当天MACD变平滑（变化很小）
            macd_becomes_smooth = abs(df['MACD_Change'].iloc[i]) <= self.smooth_threshold

            # 条件3: 绿柱缩短（柱状图为负且在增加）
            macdh_negative = df['MACDh_12_26_9'].iloc[i] < 0  # 绿柱（负值）
            macdh_shrinking = df['MACDh_Change'].iloc[i] > 0  # 缩短（负值变小）

            # 买入条件：MACD由跌转平 + 绿柱缩短
            if macd_was_falling and macd_becomes_smooth and macdh_negative and macdh_shrinking:
                df.loc[df.index[i], 'Signal'] = 1
                df.loc[df.index[i], 'Signal_Reason'] = 'MACD由跌转平+绿柱缩短'

            # ============ 卖出信号 ============
            # 条件1: 前一天MACD上升
            macd_was_rising = df['MACD_Change'].iloc[i - 1] > 0

            # 条件2: 当天MACD变平滑
            macd_becomes_smooth_sell = abs(df['MACD_Change'].iloc[i]) <= self.smooth_threshold

            # 条件3: 红柱缩短（柱状图为正且在减少）
            macdh_positive = df['MACDh_12_26_9'].iloc[i] > 0  # 红柱（正值）
            macdh_shrinking_sell = df['MACDh_Change'].iloc[i] < 0  # 缩短（正值变小）

            # 卖出条件：MACD由涨转平 + 红柱缩短
            if macd_was_rising and macd_becomes_smooth_sell and macdh_positive and macdh_shrinking_sell:
                df.loc[df.index[i], 'Signal'] = -1
                df.loc[df.index[i], 'Signal_Reason'] = 'MACD由涨转平+红柱缩短'

        return df

    def backtest(self):
        """
        执行回测

        Returns:
            dict: 回测结果
        """
        # 生成信号
        df = self.generate_signals()

        # 重置状态
        self.balance = self.initial_balance
        self.shares = 0
        self.position_open = False
        self.trades = []
        self.positions = []
        self.balance_history = []
        self.portfolio_values = []

        # 逐日执行交易
        for i in range(len(df)):
            date = df.index[i]
            price = df['Close'].iloc[i]
            signal = df['Signal'].iloc[i]

            # 买入信号
            if signal == 1 and not self.position_open:
                self._buy(date, price)

            # 卖出信号
            elif signal == -1 and self.position_open:
                self._sell(date, price)

            # 记录每日资产
            portfolio_value = self.balance + self.shares * price
            self.balance_history.append(self.balance)
            self.portfolio_values.append(portfolio_value)

        # 如果最后还有持仓，平仓
        if self.position_open:
            last_price = df['Close'].iloc[-1]
            last_date = df.index[-1]
            self._sell(last_date, last_price)

        # 计算回测指标
        metrics = self._calculate_metrics(df)

        # 添加信号数据
        self.signal_data = df[['Close', 'MACD_12_26_9', 'MACDh_12_26_9', 'Signal']].copy()

        return metrics

    def _buy(self, date, price):
        """执行买入"""
        # 使用全部资金买入
        max_shares = int(self.balance / (price * (1 + self.commission)))

        if max_shares > 0:
            cost = max_shares * price * (1 + self.commission)
            self.balance -= cost
            self.shares += max_shares
            self.position_open = True
            self.buy_price = price

            self.trades.append({
                'date': date,
                'type': 'BUY',
                'price': price,
                'shares': max_shares,
                'cost': cost,
                'balance': self.balance
            })

    def _sell(self, date, price):
        """执行卖出"""
        if self.shares > 0:
            revenue = self.shares * price * (1 - self.commission)
            profit = revenue - (self.shares * self.buy_price * (1 + self.commission))
            profit_pct = (profit / (self.shares * self.buy_price * (1 + self.commission))) * 100

            self.balance += revenue

            self.trades.append({
                'date': date,
                'type': 'SELL',
                'price': price,
                'shares': self.shares,
                'revenue': revenue,
                'balance': self.balance,
                'profit': profit,
                'profit_pct': profit_pct
            })

            self.shares = 0
            self.position_open = False
            self.buy_price = 0

    def _calculate_metrics(self, df):
        """计算回测指标"""
        final_value = self.portfolio_values[-1]

        # 总收益率
        total_return = (final_value - self.initial_balance) / self.initial_balance

        # 最大回撤
        portfolio_values = np.array(self.portfolio_values)
        running_max = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - running_max) / running_max
        max_drawdown = drawdown.min()

        # 交易统计
        buy_trades = [t for t in self.trades if t['type'] == 'BUY']
        sell_trades = [t for t in self.trades if t['type'] == 'SELL']

        # 盈利交易
        profitable_trades = [t for t in sell_trades if t.get('profit', 0) > 0]
        losing_trades = [t for t in sell_trades if t.get('profit', 0) <= 0]

        win_rate = len(profitable_trades) / len(sell_trades) if sell_trades else 0

        # 平均盈亏
        avg_profit = np.mean([t['profit'] for t in profitable_trades]) if profitable_trades else 0
        avg_loss = np.mean([t['profit'] for t in losing_trades]) if losing_trades else 0

        # 盈亏比
        profit_loss_ratio = abs(avg_profit / avg_loss) if avg_loss != 0 else 0

        # 夏普比率（简化版）
        if len(self.portfolio_values) > 1:
            returns = np.diff(self.portfolio_values) / self.portfolio_values[:-1]
            sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0
            sharpe_ratio *= np.sqrt(252)  # 年化
        else:
            sharpe_ratio = 0

        # 买入持有策略对比
        buy_hold_return = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]

        # 持仓天数
        total_days = len(df)
        position_days = sum([1 for v in self.portfolio_values if
                             v > self.balance_history[self.portfolio_values.index(v)] + 100])  # 简化计算

        metrics = {
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'final_value': final_value,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'sharpe_ratio': sharpe_ratio,
            'total_trades': len(sell_trades),
            'profitable_trades': len(profitable_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_loss_ratio': profit_loss_ratio,
            'buy_hold_return': buy_hold_return,
            'buy_hold_return_pct': buy_hold_return * 100,
            'outperformance': (total_return - buy_hold_return) * 100,
            'total_days': total_days,
        }

        return metrics

    def get_trade_details(self):
        """获取交易详情"""
        if not self.trades:
            return pd.DataFrame()

        df_trades = pd.DataFrame(self.trades)
        return df_trades

    def get_signals_summary(self):
        """获取信号汇总"""
        if not hasattr(self, 'signal_data'):
            return None

        buy_signals = self.signal_data[self.signal_data['Signal'] == 1]
        sell_signals = self.signal_data[self.signal_data['Signal'] == -1]

        return {
            'total_buy_signals': len(buy_signals),
            'total_sell_signals': len(sell_signals),
            'buy_signal_dates': buy_signals.index.tolist(),
            'sell_signal_dates': sell_signals.index.tolist()
        }


# 测试代码
if __name__ == "__main__":
    from improved_data_engine import DataEngine

    # 加载数据
    engine = DataEngine("AAPL")
    df = engine.load_processed_data()

    if df is None or df.empty:
        print("请先下载数据")
        exit(1)

    print("\n" + "=" * 60)
    print("MACD短线交易策略回测")
    print("=" * 60)

    # 创建策略
    strategy = MACDStrategy(df, initial_balance=10000, commission=0.001)

    # 执行回测
    metrics = strategy.backtest()

    # 显示结果
    print(f"\n{'=' * 60}")
    print("回测结果")
    print(f"{'=' * 60}")
    print(f"总收益率: {metrics['total_return_pct']:.2f}%")
    print(f"最终资产: ${metrics['final_value']:,.2f}")
    print(f"最大回撤: {metrics['max_drawdown_pct']:.2f}%")
    print(f"夏普比率: {metrics['sharpe_ratio']:.3f}")
    print(f"总交易次数: {metrics['total_trades']}")
    print(f"盈利交易: {metrics['profitable_trades']}")
    print(f"亏损交易: {metrics['losing_trades']}")
    print(f"胜率: {metrics['win_rate'] * 100:.2f}%")
    print(f"盈亏比: {metrics['profit_loss_ratio']:.2f}")
    print(f"\n买入持有收益: {metrics['buy_hold_return_pct']:.2f}%")
    print(f"策略超额收益: {metrics['outperformance']:.2f}%")
    print(f"{'=' * 60}\n")

    # 显示交易记录
    trades = strategy.get_trade_details()
    if not trades.empty:
        print("最近5笔交易:")
        print(trades.tail(5).to_string())

    # 显示信号汇总
    signals = strategy.get_signals_summary()
    print(f"\n买入信号数: {signals['total_buy_signals']}")
    print(f"卖出信号数: {signals['total_sell_signals']}")