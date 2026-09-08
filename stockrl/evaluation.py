"""Algorithm-independent evaluation on the same account and execution engine."""
import math
import numpy as np


def baseline_policy(name):
    """cash, buy_hold, fixed half weight, or close versus trailing 20-day mean."""
    if name not in {'cash','buy_hold','half','trend'}:
        raise ValueError(f'Unknown baseline: {name}')

    def policy(obs,env):
        if name == 'cash': return -1.
        if name == 'half': return 0.
        if name == 'buy_hold':
            # Persistent full-allocation request accumulates shares when entry
            # is liquidity-limited and invests residual cash if affordable.
            # It never intentionally sells an existing position.
            return 1.
        close = env.bars.Close.iloc[max(0,env.current_step-19):env.current_step+1]
        return 1. if close.iloc[-1] > close.mean() else -1.
    return policy


def evaluate_policy(env,policy):
    obs,_ = env.reset()
    while True:
        obs,_,terminated,truncated,_ = env.step(policy(obs,env))
        if terminated or truncated: break
    history = [dict(row) for row in env.history]
    trades = [dict(row) for row in env.trades]
    nav = np.asarray([row['nav'] for row in history],dtype=float)
    returns = nav[1:]/nav[:-1]-1
    periods = len(returns)
    volatility = float(returns.std(ddof=1)*np.sqrt(252)) if periods > 1 else None
    sharpe = float(returns.mean()*252/volatility) if volatility is not None and volatility > 1e-12 else None
    net_return = float(nav[-1]/nav[0]-1) if len(nav)>1 else None
    annualized = -1. if periods else None
    if periods and nav[-1] > 0:
        exponent = (math.log(nav[-1])-math.log(nav[0]))*252/periods
        annualized = math.expm1(exponent) if exponent <= math.log(np.finfo(float).max) else None
    metrics = {'total_return':net_return,'annualized_return':annualized,'volatility':volatility,
        'sharpe':sharpe,'max_drawdown':float(np.max(1-nav/np.maximum.accumulate(nav))),
        'total_turnover':float(sum(row['turnover'] for row in history)),
        'total_cost':float(sum(row['cost'] for row in history)),
        'average_weight':float(np.mean([row['weight'] for row in history[1:]])),
        'trade_count':len(trades)}
    # Extremely short, explosive paths can overflow annualization; undefined
    # values remain valid JSON nulls instead of Infinity/NaN.
    metrics = {key:(None if isinstance(value,float) and not math.isfinite(value) else value)
               for key,value in metrics.items()}
    return {'metrics':metrics,'history':history,'trades':trades}
