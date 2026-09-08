"""One accounting engine for learned policies and comparison baselines."""
from dataclasses import dataclass
import math
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from .data import validate_bars
from .features import build_features


@dataclass(frozen=True)
class TradingConfig:
    initial_cash: float = 10000.0
    commission: float = .001
    slippage: float = .0005
    sell_tax: float = 0.0
    min_commission: float = 0.0
    lot_size: int = 1
    max_participation: float = .01
    t_plus_one: bool = True
    drawdown_penalty: float = 0.0
    turnover_penalty: float = 0.0

    def __post_init__(self):
        for name in ('initial_cash','commission','slippage','sell_tax','min_commission',
                     'max_participation','drawdown_penalty','turnover_penalty'):
            value = getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,(int,float,np.number)) or not np.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be a finite nonnegative number')
        if self.initial_cash <= 0:
            raise ValueError('initial_cash must be positive')
        if self.slippage >= 1 or self.commission >= 1 or self.sell_tax >= 1:
            raise ValueError('slippage, commission and sell_tax must be less than one')
        if not 0 <= self.max_participation <= 1:
            raise ValueError('max_participation must be between zero and one')
        if isinstance(self.lot_size,bool) or not isinstance(self.lot_size,(int,np.integer)) or self.lot_size < 1:
            raise ValueError('lot_size must be a positive integer')
        if not isinstance(self.t_plus_one,bool):
            raise ValueError('t_plus_one must be boolean')


class TradingEnv(gym.Env):
    """Observe close t, rebalance at open t+1, then mark at close t+1.

    Target shares use the pre-trade open NAV; whole-lot rounding, liquidity and
    fees can prevent the requested weight being reached. Capacity is estimated
    from the previous 20 observed volumes. Execution-day volume is consulted
    only for the zero-volume suspension flag, never for sizing a fill.
    A daily environment has at most one rebalance per day: yesterday's purchases
    become sellable at the next open under T+1. No intraday resale is possible.
    """
    metadata = {'render_modes': []}

    def __init__(self,bars,config=None,start=None,end=None,normalizer=None,
                 random_start=False,episode_length=None):
        super().__init__()
        self.bars = validate_bars(bars)
        self.config = config or TradingConfig()
        self.features = build_features(self.bars)
        transformed = normalizer.transform(self.features) if normalizer is not None else self.features
        self._feature_values = np.clip(transformed.to_numpy(dtype=float), -1e6, 1e6).astype(np.float32)
        self.feature_names = list(self.features.columns) + ['cash_fraction','position_weight','drawdown']
        self.start = self._boundary(start,0)
        self.end = self._boundary(end,len(self.bars)-1)
        if not 0 <= self.start < self.end < len(self.bars):
            raise ValueError('Bounds require 0 <= start < end < number of bars')
        if episode_length is not None and (isinstance(episode_length,bool) or
            not isinstance(episode_length,(int,np.integer)) or episode_length < 1):
            raise ValueError('episode_length must be a positive integer')
        self.random_start = bool(random_start)
        self.episode_length = episode_length
        self.action_space = spaces.Box(-1.0,1.0,shape=(1,),dtype=np.float32)
        self.observation_space = spaces.Box(-1e6,1e6,shape=(len(self.feature_names),),dtype=np.float32)
        self.history = []
        self.trades = []
        self._done = True

    def _boundary(self,value,default):
        if value is None: return default
        if isinstance(value,(int,np.integer)) and not isinstance(value,bool): return int(value)
        try:
            location = self.bars.index.get_loc(pd.Timestamp(value))
        except (ValueError,KeyError,TypeError) as exc:
            raise ValueError(f'Boundary date not found: {value}') from exc
        return int(location)

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        length = min(self.episode_length or (self.end-self.start),self.end-self.start)
        last_start = self.end-length
        self.current_step = int(self.np_random.integers(self.start,last_start+1)) if self.random_start else self.start
        self._episode_end = self.current_step+length
        self.cash = float(self.config.initial_cash)
        self.shares = 0
        self.nav = self.cash
        self._peak = self.nav
        self._drawdown = 0.
        self._done = False
        self.trades = []
        initial = self._snapshot(0.,0.,0.,0.)
        self.history = [initial]
        return self._observation(),dict(initial)

    def _observation(self):
        weight = self.shares*float(self.bars.Close.iloc[self.current_step])/self.nav if self.nav > 0 else 0.
        account = np.array([self.cash/self.nav if self.nav > 0 else 0.,weight,self._drawdown],dtype=np.float32)
        return np.concatenate((self._feature_values[self.current_step],account)).astype(np.float32)

    def _snapshot(self,turnover,cost,requested,executed):
        return {'date':self.bars.index[self.current_step].isoformat().split('T')[0],
            'nav':float(self.nav),'cash':float(self.cash),'shares':int(self.shares),
            'weight':float(self.shares*self.bars.Close.iloc[self.current_step]/self.nav) if self.nav > 0 else 0.,
            'turnover':float(turnover),'cost':float(cost),'requested_weight':float(requested),
            'executed_weight':float(executed)}

    def _commission(self,notional):
        return max(self.config.min_commission,notional*self.config.commission) if notional > 0 else 0.

    def step(self,action):
        if self._done:
            raise RuntimeError('Episode has ended or not started; call reset before step')
        values = np.asarray(action,dtype=float)
        if values.size != 1 or not np.isfinite(values).all():
            raise ValueError('Action must be one finite scalar in [-1,1]')
        requested = (float(np.clip(values.item(),-1,1))+1)/2
        cfg = self.config
        previous_nav = self.nav
        old_drawdown = self._drawdown
        decision = self.current_step
        execution = decision+1
        open_price = float(self.bars.Open.iloc[execution])
        open_nav = self.cash+self.shares*open_price
        desired = math.floor(requested*open_nav/open_price/cfg.lot_size + 1e-12)*cfg.lot_size
        delta = desired-self.shares
        historical_volume = float(self.bars.Volume.iloc[max(0,decision-19):decision+1].mean())
        capacity = math.floor(historical_volume*cfg.max_participation/cfg.lot_size)*cfg.lot_size
        quantity = min(abs(delta),capacity)
        if float(self.bars.Volume.iloc[execution]) == 0:
            quantity = 0
        side = 'buy' if delta > 0 else 'sell'
        fill_price = open_price*(1+cfg.slippage if side == 'buy' else 1-cfg.slippage)
        if side == 'buy':
            # Both proportional and minimum fee bounds must hold simultaneously.
            affordable = min(self.cash/(fill_price*(1+cfg.commission)),
                             max(0,self.cash-cfg.min_commission)/fill_price)
            quantity = min(quantity,math.floor(affordable/cfg.lot_size + 1e-12)*cfg.lot_size)
        else:
            quantity = min(quantity,self.shares)
        fee = self._commission(quantity*fill_price)
        tax = quantity*fill_price*cfg.sell_tax if side == 'sell' else 0.
        if side == 'sell' and self.cash+quantity*fill_price-fee-tax < -1e-9:
            # An uneconomic minimum-fee sale cannot create debt.
            quantity = 0
            fee = tax = 0.
        cost = turnover = 0.
        if quantity > 0:
            notional = quantity*fill_price
            self.cash += (-notional-fee) if side == 'buy' else (notional-fee-tax)
            if -1e-8 < self.cash < 0: self.cash = 0.
            self.shares += quantity if side == 'buy' else -quantity
            cost = fee+tax+quantity*abs(fill_price-open_price)
            turnover = quantity*open_price/open_nav if open_nav > 0 else 0.
            self.trades.append({'date':self.bars.index[execution].isoformat().split('T')[0],
                'decision_date':self.bars.index[decision].isoformat().split('T')[0],
                'side':side,'shares':int(quantity),'price':float(fill_price),'open_price':open_price,
                'notional':float(notional),'commission':float(fee),'tax':float(tax),
                'slippage':float(quantity*abs(fill_price-open_price)), 'cost':float(cost)})
        post_open_nav = self.cash+self.shares*open_price
        executed = self.shares*open_price/post_open_nav if post_open_nav > 0 else 0.
        self.current_step = execution
        self.nav = float(self.cash+self.shares*float(self.bars.Close.iloc[execution]))
        self._peak = max(self._peak,self.nav)
        self._drawdown = 1-self.nav/self._peak
        terminated = self.nav <= 0
        truncated = execution >= self._episode_end and not terminated
        self._done = terminated or truncated
        reward = math.log(max(self.nav,np.finfo(float).tiny)/previous_nav)
        reward -= cfg.drawdown_penalty*max(0,self._drawdown-old_drawdown)+cfg.turnover_penalty*turnover
        info = self._snapshot(turnover,cost,requested,executed)
        info['reward'] = float(reward)
        self.history.append(info)
        return self._observation(),float(reward),bool(terminated),bool(truncated),dict(info)
