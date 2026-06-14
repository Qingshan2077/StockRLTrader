"""
订单执行模拟 — partial fill / slippage / delayed execution
"""
import numpy as np
from layers.backtest.event import OrderEvent, FillEvent


class ExecutionSimulator:
    """订单执行模拟器"""

    def __init__(self, commission: float = 0.001, slippage: float = 0.0005,
                 max_fill_ratio: float = 0.3, delay: int = 1):
        """
        Args:
            commission: 手续费率
            slippage: 滑点率
            max_fill_ratio: 单次最多成交 ADV 的比例
            delay: 延迟 bar 数 (0 = 当前 bar)
        """
        self.commission = commission
        self.slippage = slippage
        self.max_fill_ratio = max_fill_ratio
        self.delay = delay
        self._pending_orders: list[tuple[int, OrderEvent]] = []

    def submit_order(self, order: OrderEvent, current_step: int) -> None:
        """提交订单到延迟队列"""
        fill_step = current_step + self.delay
        self._pending_orders.append((fill_step, order))

    def get_fills(self, current_step: int, current_price: float,
                  avg_daily_volume: float = None) -> list[FillEvent]:
        """获取当前步应该成交的订单列表"""
        fills = []
        remaining = []

        for fill_step, order in self._pending_orders:
            if fill_step <= current_step:
                fill = self._simulate_fill(order, current_price, avg_daily_volume)
                fills.append(fill)
            else:
                remaining.append((fill_step, order))

        self._pending_orders = remaining
        return fills

    def _simulate_fill(self, order: OrderEvent, price: float,
                        adv: float = None) -> FillEvent:
        """模拟单笔订单成交"""
        # 滑点
        slip_dir = 1 if order.direction == "BUY" else -1
        fill_price = price * (1 + slip_dir * self.slippage)

        # 部分成交 (基于 ADV)
        fill_ratio = 1.0
        if adv is not None and adv > 0:
            order_value = order.quantity * price
            adv_ratio = order_value / adv
            if adv_ratio > self.max_fill_ratio:
                fill_ratio = self.max_fill_ratio / adv_ratio

        filled_qty = order.quantity * fill_ratio
        commission = filled_qty * fill_price * self.commission

        return FillEvent(
            ticker=order.ticker,
            direction=order.direction,
            quantity=filled_qty,
            fill_price=fill_price,
            commission=commission,
            slippage=abs(fill_price - price),
            fill_ratio=fill_ratio,
        )

    @property
    def pending_count(self) -> int:
        return len(self._pending_orders)
