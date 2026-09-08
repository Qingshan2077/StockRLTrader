"""Causal features and serializable train-fitted normalization."""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .data import validate_bars


def build_features(bars: pd.DataFrame) -> pd.DataFrame:
    """All rolling windows end on the observation date, including warm-up."""
    bars = validate_bars(bars)
    close = bars.Close
    returns = np.log(close).diff().fillna(0)
    volume_mean = bars.Volume.rolling(20,min_periods=1).mean()
    frame = pd.DataFrame({
        'log_return':returns,
        'intraday_return':np.log(close / bars.Open),
        'range':(bars.High - bars.Low)/close,
        'volatility_20':returns.rolling(20,min_periods=1).std(ddof=0),
        'momentum_5':np.log(close / close.shift(5).fillna(close.iloc[0])),
        'momentum_20':np.log(close / close.shift(20).fillna(close.iloc[0])),
        'volume_relative':bars.Volume / volume_mean.replace(0,np.nan) - 1,
        'trend_20':close / close.rolling(20,min_periods=1).mean() - 1,
    }, index=bars.index)
    return frame.replace([np.inf,-np.inf],0).fillna(0).astype(float)


@dataclass(frozen=True)
class ObservationNormalizer:
    columns: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    clip: float = 10.0

    @classmethod
    def fit(cls, feature_frame: pd.DataFrame):
        """Fit only the explicitly supplied training observations."""
        values = feature_frame.to_numpy(dtype=float)
        if values.size == 0 or not np.isfinite(values).all():
            raise ValueError('Normalizer training features must be nonempty and finite')
        scale = values.std(axis=0)
        scale[scale < 1e-12] = 1.0
        return cls(tuple(feature_frame.columns),values.mean(axis=0),scale)

    def transform(self, feature_frame: pd.DataFrame) -> pd.DataFrame:
        if tuple(feature_frame.columns) != self.columns:
            raise ValueError('Feature columns do not match fitted normalizer')
        values = feature_frame.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError('Features must be finite')
        normalized = np.clip((values-self.mean)/self.scale,-self.clip,self.clip)
        return pd.DataFrame(normalized,index=feature_frame.index,columns=self.columns)

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps({'columns':list(self.columns),'mean':self.mean.tolist(),
            'scale':self.scale.tolist(),'clip':self.clip},indent=2),encoding='utf-8')

    @classmethod
    def load(cls, path: str | Path):
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        columns = tuple(data['columns'])
        mean, scale = np.asarray(data['mean'],dtype=float), np.asarray(data['scale'],dtype=float)
        clip = float(data['clip'])
        if (mean.shape != (len(columns),) or scale.shape != mean.shape or
            not np.isfinite(mean).all() or not np.isfinite(scale).all() or
            (scale <= 0).any() or not np.isfinite(clip) or clip <= 0):
            raise ValueError('Invalid normalizer parameters')
        return cls(columns,mean,scale,clip)
