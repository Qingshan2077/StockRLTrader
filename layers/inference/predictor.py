"""
在线推理 — 加载最新模型 → 单步推理 → 输出 signal
"""
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime


class OnlinePredictor:
    """在线推理器"""

    def __init__(self, model_path: str, scaler_path: str = None):
        self.model = None
        self.scaler = None
        self.feature_names = []
        self._load_model(model_path)
        if scaler_path:
            self._load_scaler(scaler_path)

    def _load_model(self, path: str) -> None:
        with open(Path(path), "rb") as f:
            data = pickle.load(f)
        self.model = data.get("model")
        self.feature_names = data.get("feature_names", [])

    def _load_scaler(self, path: str) -> None:
        with open(Path(path), "rb") as f:
            self.scaler = pickle.load(f)

    def predict(self, features: np.ndarray) -> dict:
        """
        单步推理

        Args:
            features: 特征向量 (1D 或 2D), 需与 training 时 feature_names 顺序一致

        Returns:
            {"signal_score": float, "timestamp": str}
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)

        if self.scaler is not None:
            features = self.scaler.transform(features)

        raw_pred = self.model.predict(features)
        if hasattr(raw_pred, '__iter__'):
            raw_pred = float(raw_pred[0])

        return {
            "signal_score": float(np.tanh(raw_pred)),
            "raw_prediction": float(raw_pred),
            "timestamp": datetime.now().isoformat(),
        }

    def __repr__(self) -> str:
        return f"OnlinePredictor(features={len(self.feature_names)})"
