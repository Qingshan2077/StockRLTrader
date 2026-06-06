"""
回测引擎 — 事件驱动 + Walk-Forward + 报告
"""
from .backtest_engine import BacktestEngine
from .event import Event, MarketEvent, SignalEvent, OrderEvent, FillEvent, EventType
from .event_queue import EventQueue
from .execution_simulator import ExecutionSimulator
from .walk_forward import WalkForwardBacktest, WalkForwardResult
from .report import BacktestReport

__all__ = [
    "BacktestEngine",
    "Event", "MarketEvent", "SignalEvent", "OrderEvent", "FillEvent", "EventType",
    "EventQueue", "ExecutionSimulator",
    "WalkForwardBacktest", "WalkForwardResult",
    "BacktestReport",
]
