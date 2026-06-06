"""
多通道日志系统

四个独立日志通道:
    - training: 模型训练日志
    - backtest: 回测执行日志
    - execution: 交易执行日志
    - risk: 风险事件日志

每条日志带: timestamp / module / level / message / extra_context
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


class LoggingSystem:
    """多通道日志管理器"""

    CHANNELS = ["training", "backtest", "execution", "risk"]

    def __init__(self, log_dir: str = "logs", level: int = logging.INFO):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.level = level
        self._loggers: dict[str, logging.Logger] = {}
        self._init_all_channels()

    def _init_all_channels(self) -> None:
        """初始化所有通道"""
        for channel in self.CHANNELS:
            self._loggers[channel] = self._create_channel(channel)

        # 根日志器 (汇总所有)
        self._loggers["system"] = self._create_channel("system")

    def _create_channel(self, name: str) -> logging.Logger:
        """创建一个日志通道 (控制台 + 文件)"""
        logger = logging.getLogger(f"Quant.{name}")
        logger.setLevel(self.level)
        logger.handlers.clear()

        # 文件 handler
        date_str = datetime.now().strftime("%Y%m%d")
        file_handler = logging.FileHandler(
            self.log_dir / f"{name}_{date_str}.log",
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)

        if name == "risk":
            fmt = "%(asctime)s | %(levelname)-7s | %(message)s"
        else:
            fmt = "%(asctime)s | %(levelname)-7s | %(module)-15s | %(message)s"

        file_handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(file_handler)

        # 控制台 handler (仅 system 和 risk 通道)
        if name in ("system", "risk"):
            console = logging.StreamHandler(sys.stdout)
            console.setLevel(self.level)
            console.setFormatter(logging.Formatter(
                "[%(levelname)-5s] %(message)s"
            ))
            logger.addHandler(console)

        return logger

    # ---------- 便捷方法 ----------

    def training(self, msg: str, **kwargs) -> None:
        self._log("training", msg, logging.INFO, kwargs)

    def backtest(self, msg: str, **kwargs) -> None:
        self._log("backtest", msg, logging.INFO, kwargs)

    def execution(self, msg: str, **kwargs) -> None:
        self._log("execution", msg, logging.INFO, kwargs)

    def risk(self, msg: str, **kwargs) -> None:
        self._log("risk", msg, logging.WARNING, kwargs)

    def info(self, msg: str, **kwargs) -> None:
        self._log("system", msg, logging.INFO, kwargs)

    def warning(self, msg: str, **kwargs) -> None:
        self._log("system", msg, logging.WARNING, kwargs)

    def error(self, msg: str, **kwargs) -> None:
        self._log("system", msg, logging.ERROR, kwargs)

    def debug(self, msg: str, **kwargs) -> None:
        self._log("system", msg, logging.DEBUG, kwargs)

    def _log(self, channel: str, msg: str,
             level: int, extra: dict) -> None:
        """内部日志方法, 构造额外上下文"""
        logger = self._loggers.get(channel)
        if logger is None:
            return

        if extra:
            context = " | ".join(f"{k}={v}" for k, v in extra.items())
            msg = f"{msg}  [{context}]"

        logger.log(level, msg)

    def close(self) -> None:
        """关闭所有日志 handler"""
        for logger in self._loggers.values():
            for handler in logger.handlers:
                handler.close()
            logger.handlers.clear()
