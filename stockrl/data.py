"""Validated, offline-first daily OHLCV data.

Prices are preserved as provided: callers are responsible for a consistent
adjustment convention and corporate-action handling in the source data.
"""
from pathlib import Path
import numpy as np
import pandas as pd

COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']


def validate_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Return a numeric copy; reject ambiguous chronology and invalid OHLCV."""
    if not isinstance(bars, pd.DataFrame):
        raise ValueError('OHLCV input must be a pandas DataFrame')
    missing = set(COLUMNS) - set(bars.columns)
    if missing:
        raise ValueError(f'Missing OHLCV columns: {sorted(missing)}')
    if bars.empty:
        raise ValueError('OHLCV data must contain at least one row')
    if not isinstance(bars.index, pd.DatetimeIndex):
        raise ValueError('Date index must be a DatetimeIndex')
    if bars.index.hasnans or not bars.index.is_unique or not bars.index.is_monotonic_increasing:
        raise ValueError('Date values must be valid, unique and strictly increasing')
    if not bars.index.normalize().is_unique:
        raise ValueError('Daily OHLCV permits only one observation per calendar day')
    result = bars[COLUMNS].copy()
    try:
        result = result.apply(pd.to_numeric, errors='raise').astype(float)
    except (ValueError, TypeError) as exc:
        raise ValueError('OHLCV values must be numeric and finite') from exc
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError('OHLCV values must be finite (no NaN or infinity)')
    if (result[['Open','High','Low','Close']] <= 0).any().any():
        raise ValueError('OHLC prices must be positive')
    if (result.Volume < 0).any():
        raise ValueError('Volume must be nonnegative')
    invalid = ((result.High < result[['Open','Close','Low']].max(axis=1)) |
               (result.Low > result[['Open','Close','High']].min(axis=1)))
    if invalid.any():
        raise ValueError(f'Invalid OHLC price range at Date {result.index[invalid][0]}')
    result.index.name = 'Date'
    return result


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load Date/Open/High/Low/Close/Volume; ignore additional columns."""
    try:
        frame = pd.read_csv(path, float_precision='round_trip')
    except pd.errors.EmptyDataError as exc:
        raise ValueError('CSV must contain Date and OHLCV columns') from exc
    if 'Date' not in frame:
        raise ValueError('CSV is missing Date column')
    try:
        dates = pd.to_datetime(frame.pop('Date'), errors='raise')
        frame.index = pd.DatetimeIndex(dates, name='Date')
    except (ValueError, TypeError) as exc:
        raise ValueError('Date column contains invalid or mixed-timezone dates') from exc
    return validate_bars(frame)


def make_demo_data(n: int = 756, seed: int = 42) -> pd.DataFrame:
    """Synthetic data for software demonstrations, never evidence of returns."""
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 2:
        raise ValueError('Demo data requires an integer n >= 2')
    rng = np.random.default_rng(seed)
    returns = rng.normal(.00025, .014, n)
    close = 100 * np.exp(np.cumsum(returns))
    previous = np.r_[100., close[:-1]]
    open_ = previous * np.exp(rng.normal(0, .004, n))
    spread = rng.uniform(.002, .018, n)
    frame = pd.DataFrame({'Open':open_, 'High':np.maximum(open_,close)*(1+spread),
        'Low':np.minimum(open_,close)*(1-spread), 'Close':close,
        'Volume':rng.integers(100000,1000000,n).astype(float)},
        index=pd.bdate_range('2020-01-02',periods=n,name='Date'))
    return validate_bars(frame)
