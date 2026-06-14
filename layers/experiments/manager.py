"""
实验管理器 — 追踪实验配置/结果/模型
"""
import hashlib
import json
from datetime import datetime
from pathlib import Path
from layers.experiments.storage import ExperimentStorage


class ExperimentManager:
    """实验追踪管理器"""

    def __init__(self, storage: ExperimentStorage = None):
        self.storage = storage or ExperimentStorage()

    def create_experiment(self, name: str, config: dict,
                          tags: list[str] = None) -> str:
        """创建新实验, 返回实验 ID"""
        config_json = json.dumps(config, sort_keys=True, default=str)
        config_hash = hashlib.md5(config_json.encode()).hexdigest()[:12]
        exp_id = f"{datetime.now():%Y%m%d_%H%M%S}_{config_hash}"

        self.storage.save_experiment({
            "exp_id": exp_id,
            "name": name,
            "config": config,
            "config_hash": config_hash,
            "tags": tags or [],
            "created_at": datetime.now().isoformat(),
            "status": "running",
        })
        return exp_id

    def log_metrics(self, exp_id: str, metrics: dict,
                    step: int = None) -> None:
        """记录指标"""
        self.storage.save_metrics(exp_id, {
            "step": step,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        })

    def log_artifact(self, exp_id: str, name: str, path: str) -> None:
        """记录产出物 (模型文件等)"""
        self.storage.save_artifact(exp_id, {
            "name": name,
            "path": str(path),
            "timestamp": datetime.now().isoformat(),
        })

    def complete_experiment(self, exp_id: str, final_metrics: dict = None,
                            status: str = "completed") -> None:
        """标记实验完成"""
        self.storage.update_experiment(exp_id, {
            "status": status,
            "completed_at": datetime.now().isoformat(),
            "final_metrics": final_metrics or {},
        })

    def list_experiments(self, status: str = None,
                         tag: str = None) -> list[dict]:
        """查询实验列表"""
        return self.storage.query_experiments(status=status, tag=tag)

    def compare(self, exp_ids: list[str],
                metrics: list[str] = None) -> list[dict]:
        """对比多个实验"""
        return self.storage.compare_experiments(exp_ids, metrics)

    def get_best(self, metric: str = "sharpe_ratio",
                 top_n: int = 5) -> list[dict]:
        """获取指定指标最优的 N 个实验"""
        all_exps = self.storage.query_experiments()
        ranked = []
        for exp in all_exps:
            fm = exp.get("final_metrics", {})
            if metric in fm:
                ranked.append((fm[metric], exp))
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [exp for _, exp in ranked[:top_n]]

    def __repr__(self) -> str:
        return f"ExperimentManager(storage={self.storage})"
