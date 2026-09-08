import pytest
import pandas as pd
from stockrl.data import make_demo_data
from stockrl.env import TradingEnv, TradingConfig
from stockrl.evaluation import baseline_policy, evaluate_policy


def test_baseline_path_uses_same_next_open_account():
    frame = pd.DataFrame({'Open':[100,200,300], 'High':[100,220,330],
        'Low':[100,200,300], 'Close':[100,220,330], 'Volume':[100000]*3},
        index=pd.date_range('2024-01-01', periods=3, name='Date'))
    env = TradingEnv(frame, TradingConfig(commission=0, slippage=0,max_participation=1))
    result = evaluate_policy(env, baseline_policy('buy_hold'))
    assert [row['nav'] for row in result['history']] == [10000,11000,16500]
    assert result['metrics']['total_return'] == pytest.approx(.65)
    assert result['metrics']['trade_count'] == 1
    assert result['metrics']['total_turnover'] == 1
    assert result['metrics']['max_drawdown'] == 0


def test_cash_metrics_no_volatility_sharpe_undefined():
    env = TradingEnv(make_demo_data(n=10))
    result = evaluate_policy(env, baseline_policy('cash'))
    assert result['metrics']['total_return'] == 0
    assert result['metrics']['sharpe'] is None
    assert result['metrics']['volatility'] == 0
    assert result['metrics']['total_cost'] == 0
    assert result['trades'] == []


def test_half_and_trend_are_callable_and_unknown_rejected():
    env = TradingEnv(make_demo_data(n=30))
    obs, _ = env.reset()
    assert baseline_policy('half')(obs, env) == 0
    assert -1 <= baseline_policy('trend')(obs, env) <= 1
    with pytest.raises(ValueError): baseline_policy('magic')


def test_buy_hold_does_not_sell_after_gap_when_rounding_left_cash():
    frame = pd.DataFrame({'Open':[60,60,100], 'High':[60,60,100],
        'Low':[60,60,100], 'Close':[60,60,100], 'Volume':[100000]*3},
        index=pd.date_range('2024-01-01', periods=3, name='Date'))
    env = TradingEnv(frame, TradingConfig(commission=0, slippage=0,max_participation=1))
    result = evaluate_policy(env, baseline_policy('buy_hold'))
    assert [row['shares'] for row in result['history']] == [0,166,166]
    assert len(result['trades']) == 1


def test_buy_hold_preserves_million_shares_despite_float_roundoff():
    frame = pd.DataFrame({'Open':[.01,.01,.07], 'High':[.01,.01,.07],
        'Low':[.01,.01,.07], 'Close':[.01,.01,.07], 'Volume':[100000000]*3},
        index=pd.date_range('2024-01-01', periods=3, name='Date'))
    env = TradingEnv(frame,TradingConfig(commission=0,slippage=0,max_participation=1))
    result = evaluate_policy(env,baseline_policy('buy_hold'))
    assert [row['shares'] for row in result['history']] == [0,1000000,1000000]
    assert len(result['trades']) == 1
    assert result['history'][-1]['cash'] == 0


def test_explosive_single_period_annualization_is_null_without_warning():
    import warnings
    frame = pd.DataFrame({'Open':[100,100], 'High':[100,2000],
        'Low':[100,100], 'Close':[100,2000], 'Volume':[100000]*2},
        index=pd.date_range('2024-01-01', periods=2, name='Date'))
    env = TradingEnv(frame,TradingConfig(commission=0,slippage=0,max_participation=1))
    with warnings.catch_warnings():
        warnings.simplefilter('error',RuntimeWarning)
        result = evaluate_policy(env,baseline_policy('buy_hold'))
    assert result['metrics']['total_return'] == 19
    assert result['metrics']['annualized_return'] is None
    assert result['metrics']['volatility'] is None
    assert result['metrics']['sharpe'] is None
