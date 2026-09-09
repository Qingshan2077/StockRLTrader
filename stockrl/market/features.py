"""Causal session prices and train-only normalization for protocol v2."""
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from stockrl.features import ObservationNormalizer


def known(available_at, at):
    return datetime.fromisoformat(available_at) <= datetime.fromisoformat(at)


def actions_for(bundle, session, at):
    return tuple(a for a in bundle.actions if a.effective_session == session and known(a.available_at,at))


def split_factor(actions):
    result=1
    for action in actions:
        if action.kind=='split':
            ratio=action.split_ratio
            if ratio is None or not float(ratio).is_integer() or ratio<1:
                raise ValueError('CORPORATE_ACTION_UNSUPPORTED')
            result*=int(ratio)
    return result


def valuation_close(bundle, session):
    """Resolve only the latest authentic quote or audited action reference."""
    bars={b.session:b for b in bundle.bars if b.session<=session}
    statuses={s.session:s for s in bundle.tradability}
    for item in reversed(bundle.sessions):
        if not item.is_open or item.session>session:
            continue
        if item.session in bars:
            return float(bars[item.session].close)
        if actions_for(bundle,item.session,item.open_at):
            status=statuses.get(item.session)
            if not status or status.reference_close is None or not bundle.metadata.get('suspension_valuation_source') or not known(status.available_at,item.open_at):
                raise ValueError('SUSPENSION_ACTION_VALUATION_MISSING')
            return float(status.reference_close)
    raise ValueError('INITIAL_VALUATION_MISSING')


def session_view(bundle):
    """Separate authentic quotes from stale valuation; never manufacture bars."""
    bars={b.session:b for b in bundle.bars}
    statuses={s.session:s for s in bundle.tradability}
    rows=[]
    last=None
    for session in bundle.sessions:
        if not session.is_open:
            continue
        bar=bars.get(session.session)
        if bar:
            last=bar.close
        else:
            actions=actions_for(bundle,session.session,session.open_at)
            if actions:
                status=statuses.get(session.session)
                if not status or status.reference_close is None or not bundle.metadata.get('suspension_valuation_source') or not known(status.available_at,session.open_at):
                    raise ValueError('SUSPENSION_ACTION_VALUATION_MISSING')
                last=status.reference_close
            if last is None:
                raise ValueError('INITIAL_VALUATION_MISSING')
        rows.append({'session':session.session,'Open':bar.open if bar else np.nan,
                     'High':bar.high if bar else last,'Low':bar.low if bar else last,
                     'Close':last,'Volume':bar.volume if bar else 0.,'stale':bar is None})
    return pd.DataFrame(rows).set_index(pd.DatetimeIndex([r['session'] for r in rows])).drop(columns='session')


def build_feature_prices(bundle):
    view=session_view(bundle)
    sessions=[s for s in bundle.sessions if s.is_open]
    closes=[]
    for i,session in enumerate(sessions):
        actions=actions_for(bundle,session.session,session.close_at)
        split=split_factor(actions)
        dividend=sum(a.cash_per_old_share or 0 for a in actions if a.kind=='cash_dividend')
        closes.append(100. if i==0 else closes[-1]*(split*view.Close.iloc[i]+dividend)/view.Close.iloc[i-1])
    result=view[['Open','High','Low','Close','Volume']].copy()
    factor=np.asarray(closes)/view.Close.to_numpy()
    for col in ('Open','High','Low','Close'):
        result[col]=view[col].fillna(view.Close)*factor
    return result


def historical_volume(bundle, decision_session, execution_session=None, window=20):
    sessions=[s for s in bundle.sessions if s.is_open and s.session<=decision_session][-window:]
    bars={b.session:b for b in bundle.bars}
    cutoff=next(s for s in bundle.sessions if s.session==(execution_session or decision_session))
    at=cutoff.open_at if execution_session else cutoff.close_at
    volumes=[]
    for session in sessions:
        factor=split_factor(a for a in bundle.actions if session.session<a.effective_session<=cutoff.session and known(a.available_at,at))
        volumes.append((bars[session.session].volume if session.session in bars else 0)*factor)
    return float(np.mean(volumes)) if volumes else 0.


def historical_capacity(bundle, decision_session, execution_session, profile=None):
    profile=(profile or bundle.market_profile).for_session(execution_session)
    return math.floor(historical_volume(bundle,decision_session,execution_session)*profile.participation_cap)


def build_features(bundle):
    from stockrl.features import build_features as original_features
    prices=build_feature_prices(bundle)
    result=original_features(prices)
    sessions=[s for s in bundle.sessions if s.is_open]
    result['volume_relative']=[prices.Volume.iloc[i]/mean-1 if (mean:=historical_volume(bundle,s.session)) else 0 for i,s in enumerate(sessions)]
    return result


@dataclass(frozen=True)
class NormalizerV2(ObservationNormalizer):
    input_fingerprint: str = ''

    @classmethod
    def fit(cls, feature_frame, input_fingerprint=''):
        fit=ObservationNormalizer.fit(feature_frame)
        fingerprint=input_fingerprint or hashlib.sha256(pd.util.hash_pandas_object(feature_frame,index=True).values.tobytes()).hexdigest()
        return cls(fit.columns,fit.mean,fit.scale,fit.clip,fingerprint)

    def save(self,path):
        super().save(path)
        data=json.loads(Path(path).read_text(encoding='utf-8'))
        data.update(input_fingerprint=self.input_fingerprint,protocol_version=2)
        Path(path).write_text(json.dumps(data,indent=2),encoding='utf-8')

    @classmethod
    def load(cls,path):
        data=json.loads(Path(path).read_text(encoding='utf-8'))
        if data.get('protocol_version')!=2 or not data.get('input_fingerprint'):
            raise ValueError('PROTOCOL_INCOMPATIBLE')
        fit=ObservationNormalizer.load(path)
        return cls(fit.columns,fit.mean,fit.scale,fit.clip,data['input_fingerprint'])
