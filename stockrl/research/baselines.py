"""Locked transparent references. A None target means genuinely no order."""
from __future__ import annotations
import hashlib
import json
import math
from datetime import date
import numpy as np

REFERENCE_IDS = ('cash','buy_hold','fixed_25','fixed_50','fixed_75','weekly_50','trend_20','vol_target_10')


def reference_weight(strategy_id, feature_closes, execution_session, previous_session, *, matched_weight=None):
    if strategy_id == 'cash': return 0.
    if strategy_id == 'buy_hold': return 1.
    if strategy_id in ('fixed_25','fixed_50','fixed_75'):
        return int(strategy_id.split('_')[1])/100
    if strategy_id == 'matched_fixed':
        if matched_weight is None or not 0 <= matched_weight <= 1: raise ValueError('locked matched weight required')
        return float(matched_weight)
    if strategy_id == 'weekly_50':
        week = lambda s: date.fromisoformat(s).isocalendar()[:2]
        return .5 if week(execution_session) != week(previous_session) else None
    closes = np.asarray(feature_closes,dtype=float)
    if strategy_id == 'trend_20':
        return float(len(closes)>=20 and closes[-1]>closes[-20:].mean())
    if strategy_id == 'vol_target_10':
        if len(closes)<21: return 0.
        returns=closes[-20:]/closes[-21:-1]-1
        sigma=float(np.std(returns,ddof=1)*np.sqrt(252))
        return min(1.,.10/max(sigma,.01))
    raise ValueError(f'unknown reference: {strategy_id}')


def match_fixed_weight(rl_volatility, evaluate_volatility):
    if rl_volatility is None or not math.isfinite(rl_volatility):
        raise ValueError('validation volatility is unavailable')
    rows=[]
    for i in range(21):
        w=i/20
        volatility=evaluate_volatility(w)
        if volatility is None or not math.isfinite(volatility): raise ValueError('reference validation volatility unavailable')
        rows.append(dict(weight=w,volatility=volatility,calibration_error=abs(volatility-rl_volatility)))
    minimum=min(r['calibration_error'] for r in rows)
    # Decimal grid ties should not be broken by binary floating-point noise.
    selected=next(r for r in rows if math.isclose(r['calibration_error'],minimum,rel_tol=1e-12,abs_tol=1e-14))
    return {**selected,'candidates':rows}


def baseline_cache_key(*,dataset_fingerprint,market_profile,window,strategy_id,initial_cash=10000,seed=None,matched_weight=None,protocol_fingerprint=None):
    from stockrl.market.contracts import to_dict
    payload=dict(engine_version=2,dataset_fingerprint=dataset_fingerprint,market_profile=to_dict(market_profile),
                 window=window.model_dump(mode='json') if hasattr(window,'model_dump') else window,
                 strategy_id=strategy_id,initial_cash=initial_cash,protocol_fingerprint=protocol_fingerprint)
    if strategy_id=='matched_fixed': payload.update(seed=seed,matched_weight=matched_weight)
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
