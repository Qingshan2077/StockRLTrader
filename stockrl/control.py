"""Optional cooperative controls; no web, persistence or learner dependencies."""

from collections.abc import Callable
from dataclasses import dataclass, field
from time import monotonic
from typing import Any


class ExperimentCancelled(RuntimeError):
    """Execution stopped before a complete research result was produced."""


@dataclass
class ExecutionControl:
    """Check at safe boundaries and report observational events only.

    Raising an exception aborts learning without interpreting a partially trained
    model as a selected checkpoint. The owner of this object owns cancellation;
    the research engine does not read databases or manage processes.
    """

    cancel_requested: Callable[[], bool] = lambda: False
    event_callback: Callable[[dict[str, Any]], None] | None = None
    progress_interval: float = 1.0
    _last_progress: float = field(default=float("-inf"), init=False)

    def check(self) -> None:
        if self.cancel_requested():
            raise ExperimentCancelled("实验已取消或执行协调进程已断开。")

    def emit(self, event: dict[str, Any], *, progress: bool = False) -> None:
        self.check()
        now = monotonic()
        if progress and now - self._last_progress < self.progress_interval:
            return
        if progress:
            self._last_progress = now
        if self.event_callback is not None:
            self.event_callback(event)

    def phase(self, phase: str, **context: Any) -> None:
        self.emit({"event": "phase", "phase": phase, **context})
