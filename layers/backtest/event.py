"""
事件驱动回测 — 事件类型定义

MarketEvent → SignalEvent → OrderEvent → FillEvent
"""
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime


class EventType(Enum):
    MARKET = "MARKET"         # 市场数据到达
    SIGNAL = "SIGNAL"         # 交易信号
    ORDER = "ORDER"           # 订单提交
    FILL = "FILL"             # 订单成交
    STOP_LOSS = "STOP_LOSS"   # 止损触发


@dataclass
class Event:
    """事件基类"""
    type: EventType
    timestamp: datetime = field(default_factory=datetime.now)
    ticker: str = ""


@dataclass
class MarketEvent(Event):
    """市场数据事件"""
    type: EventType = EventType.MARKET
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0


@dataclass
class SignalEvent(Event):
    """交易信号事件"""
    type: EventType = EventType.SIGNAL
    signal_score: float = 0.0
    signal_confidence: float = 0.0


@dataclass
class OrderEvent(Event):
    """订单事件"""
    type: EventType = EventType.ORDER
    direction: str = "BUY"     # BUY | SELL
    order_type: str = "MKT"    # MKT | LMT
    quantity: float = 0.0
    price: float = 0.0
    target_position: float = 0.0


@dataclass
class FillEvent(Event):
    """成交事件"""
    type: EventType = EventType.FILL
    direction: str = "BUY"
    quantity: float = 0.0
    fill_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    fill_ratio: float = 1.0  # 成交比例 (1.0 = 全部成交)


@dataclass
class StopLossEvent(Event):
    """止损触发事件"""
    type: EventType = EventType.STOP_LOSS
    reason: str = ""
    current_pnl: float = 0.0
    position_closed: float = 0.0
