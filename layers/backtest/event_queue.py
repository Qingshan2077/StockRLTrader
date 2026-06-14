"""
优先级事件队列
"""
from heapq import heappush, heappop
from layers.backtest.event import Event


class EventQueue:
    """按时间戳排序的事件队列"""

    def __init__(self):
        self._queue: list[tuple[int, Event]] = []
        self._counter = 0

    def push(self, event: Event) -> None:
        """添加事件"""
        heappush(self._queue, (self._counter, event))
        self._counter += 1

    def pop(self) -> Event | None:
        """取出最早的事件"""
        if self._queue:
            return heappop(self._queue)[1]
        return None

    def peek(self) -> Event | None:
        """查看最早的事件 (不移除)"""
        if self._queue:
            return self._queue[0][1]
        return None

    def is_empty(self) -> bool:
        return len(self._queue) == 0

    def __len__(self) -> int:
        return len(self._queue)

    def clear(self) -> None:
        self._queue.clear()
        self._counter = 0
