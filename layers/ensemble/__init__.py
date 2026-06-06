"""
Signal Ensemble — 多模型融合框架
"""
from .base_ensemble import BaseEnsemble
from .weighted_ensemble import WeightedEnsemble
from .stacking_ensemble import StackingEnsemble
from .voting_ensemble import VotingEnsemble
from .regime_ensemble import RegimeEnsemble
from .ensemble_manager import EnsembleManager

__all__ = [
    "BaseEnsemble", "WeightedEnsemble", "StackingEnsemble",
    "VotingEnsemble", "RegimeEnsemble", "EnsembleManager",
]
