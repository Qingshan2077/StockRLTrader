"""
Alpha 信号层 — 监督学习/深度学习模型

支持:
    - 传统模型: Linear, Ridge, Lasso, XGBoost, LightGBM
    - 深度学习: MLP, LSTM, GRU, TCN, Transformer
    - 统一接口: fit / predict / predict_proba / get_importance / save / load
"""
from .base_model import BaseAlphaModel
from .trainer import ModelTrainer
from .labels.label_factory import LabelFactory

__all__ = ["BaseAlphaModel", "ModelTrainer", "LabelFactory"]
