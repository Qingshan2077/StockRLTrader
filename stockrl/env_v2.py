"""Gymnasium protocol-v2 environment sharing the corporate-action ledger."""
import hashlib
import math
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stockrl.market.accounting import advance_session
from stockrl.market.contracts import AccountState, DecisionIntent, to_dict
from stockrl.market.features import build_features,build_feature_prices,session_view

_PREPARED_VIEWS=[]


def _prepared_views(bundle):
    # Bundle dataclasses are immutable. Hold identity, never trust a caller-supplied
    # fingerprint alone, and bound resident history across long worker lifetimes.
    for original,views in _PREPARED_VIEWS:
        if original is bundle:
            return tuple(frame.copy(deep=True) for frame in views)
    views=(build_feature_prices(bundle),build_features(bundle),session_view(bundle))
    _PREPARED_VIEWS.append((bundle,views))
    del _PREPARED_VIEWS[:-4]
    return tuple(frame.copy(deep=True) for frame in views)


class TradingEnvV2(gym.Env):
    metadata={'render_modes':[]}
    protocol_version=2

    def __init__(self,bundle,window,normalizer=None,initial_cash=10000,episode_length=None):
        super().__init__()
        if not math.isfinite(initial_cash) or initial_cash<=0:
            raise ValueError('initial_cash must be positive')
        if episode_length is not None and (type(episode_length) is not int or episode_length<1):
            raise ValueError('episode_length must be positive integer')
        self.bundle=bundle
        self.sessions=tuple(s for s in bundle.sessions if s.is_open)
        dates=tuple(s.session for s in self.sessions)
        rewards=tuple(window.reward_sessions if hasattr(window,'reward_sessions') else window)
        if not rewards or rewards[0] not in dates or rewards[-1] not in dates:
            raise ValueError('window must contain reward sessions')
        self.start=dates.index(rewards[0])-1
        self.end=dates.index(rewards[-1])
        if self.start<0 or dates[self.start+1:self.end+1]!=rewards:
            raise ValueError('window must be contiguous and have initial observation')
        if hasattr(window,'initial_session') and window.initial_session!=dates[self.start]:
            raise ValueError('window initial session mismatch')
        self.initial_cash=float(initial_cash)
        self.episode_length=episode_length
        self.feature_prices,self.features,self.valuation=_prepared_views(bundle)
        self.normalizer=normalizer
        transformed=normalizer.transform(self.features) if normalizer else self.features
        self._feature_values=np.clip(transformed.to_numpy(),-1e6,1e6).astype(np.float32)
        self.feature_names=list(self.features.columns)+['cash_fraction','position_weight','receivable_fraction','drawdown']
        self.action_space=spaces.Box(-1.,1.,shape=(1,),dtype=np.float32)
        self.observation_space=spaces.Box(-1e6,1e6,shape=(12,),dtype=np.float32)
        self._done=True
        self.history=[]
        self.trades=[]
        self.decision_logs=[]

    @property
    def current_session(self):
        return self.sessions[self.current_step].session

    @property
    def cash(self): return self.state.cash

    @property
    def shares(self): return self.state.shares

    @property
    def nav(self): return self.state.nav

    def _account(self):
        nav=self.state.nav
        recv=sum(self.state.receivables.values())
        weight=self.state.shares*float(self.valuation.Close.iloc[self.current_step])/nav if nav>0 else 0
        return [self.state.cash/nav if nav>0 else 0,weight,recv/nav if nav>0 else 0,1-nav/self.state.peak_nav]

    def _observation(self):
        return np.concatenate([self._feature_values[self.current_step],self._account()]).astype(np.float32)

    def _snapshot(self):
        cash,weight,recv,drawdown=self._account()
        return dict(date=self.current_session,session=self.current_session,nav=self.nav,cash=self.cash,shares=self.shares,
                    receivables=sum(self.state.receivables.values()),weight=weight,position_weight_close=weight,
                    cash_fraction=cash,receivable_fraction=recv,drawdown=drawdown,stale=bool(self.valuation.stale.iloc[self.current_step]),
                    turnover=0.,cost=0.,fees=0.,requested_weight=None,executed_weight=weight)

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        length=min(self.episode_length or self.end-self.start,self.end-self.start)
        self.current_step=int(self.np_random.integers(self.start,self.end-length+1)) if self.episode_length else self.start
        self._episode_end=self.current_step+length
        self.state=AccountState(self.initial_cash,0,0,{},self.initial_cash,self.initial_cash)
        self._done=False
        self.history=[self._snapshot()]
        self.trades=[]
        self.decision_logs=[]
        return self._observation(),dict(self.history[0])

    def step(self,action):
        if self._done:
            raise RuntimeError('call reset before step')
        decision=self.sessions[self.current_step]
        execution=self.sessions[self.current_step+1]
        raw=clipped=None
        if isinstance(action,DecisionIntent):
            intent=action
        elif action is None:
            intent=None
        else:
            values=np.asarray(action,dtype=float)
            if values.size!=1 or not np.isfinite(values).all():
                raise ValueError('action must be finite scalar')
            raw=float(values.item())
            clipped=float(np.clip(raw,-1,1))
            fingerprint=hashlib.sha256(self._observation().tobytes()).hexdigest()
            intent=DecisionIntent((clipped+1)/2,decision.session,decision.close_at,fingerprint)
        prior_nav=self.nav
        outcome=advance_session(self.state,intent,execution.session,self.bundle)
        self.state=outcome.state
        self.current_step+=1
        info=self._snapshot()
        fees=cost=turnover=0.
        quote=next((b for b in self.bundle.bars if b.session==execution.session),None)
        for fill in outcome.fills:
            fee=fill.commission+fill.sell_tax+fill.other_fees
            slip=abs(fill.price-quote.open)*fill.filled_qty
            fees+=fee
            cost+=fee+slip
            turnover+=fill.filled_qty*quote.open/prior_nav
            self.trades.append(dict(to_dict(fill),date=execution.session,decision_date=decision.session,
                                    shares=fill.filled_qty,open_price=quote.open,slippage=slip,cost=fee+slip))
        open_mark=quote.open if quote else float(self.valuation.Close.iloc[self.current_step])
        open_nav=self.cash+self.shares*open_mark+sum(self.state.receivables.values())
        info.update(fees=fees,cost=cost,turnover=turnover,requested_weight=intent.target_weight if intent else None,
                    executed_weight=self.shares*open_mark/open_nav if open_nav>0 else 0,reward=outcome.reward)
        record=dict(info,decision_at=decision.close_at,execution_at=execution.open_at,decision_session=decision.session,
                    raw_action=raw,clipped_action=clipped,raw_action_reason=None if raw is not None else 'NO_POLICY_SCALAR',
                    input_fingerprint=intent.input_fingerprint if intent else hashlib.sha256(self._feature_values[self.current_step-1].tobytes()).hexdigest(),
                    requested_qty=outcome.decision_record.requested_qty,filled_qty=outcome.decision_record.filled_qty,
                    no_fill_reason=outcome.decision_record.no_fill_reason,seed=None,fold_id=None,checkpoint_id=None)
        self.decision_logs.append(record)
        self.history.append(info)
        terminated=self.nav<=0
        truncated=self.current_step>=self._episode_end and not terminated
        self._done=terminated or truncated
        return self._observation(),outcome.reward,terminated,truncated,dict(info)


def environment_class(version):
    if version==2:
        return TradingEnvV2
    if version==1:
        from stockrl.env import TradingEnv
        return TradingEnv
    raise ValueError('PROTOCOL_INCOMPATIBLE')
