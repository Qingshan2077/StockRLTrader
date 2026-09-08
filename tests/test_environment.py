import numpy as np
import pandas as pd
import pytest
from stockrl.env import TradingConfig, TradingEnv


def bars(opens=(100,100,100,100), closes=None, volumes=None):
    closes = opens if closes is None else closes
    return pd.DataFrame({'Open': opens, 'High': np.maximum(opens, closes),
        'Low': np.minimum(opens, closes), 'Close': closes,
        'Volume': [100000]*len(opens) if volumes is None else volumes},
        index=pd.date_range('2024-01-01', periods=len(opens), name='Date'))


def config(**kwargs):
    return TradingConfig(commission=0, slippage=0, max_participation=1, **kwargs)


def test_flat_buy_sell_reinvest_conserves_cash():
    env = TradingEnv(bars(), config())
    env.reset()
    assert env.step(np.array([1], dtype=np.float32))[4]['shares'] == 100
    assert env.step(-1)[4]['cash'] == 10000
    _, reward, terminated, truncated, info = env.step(1)
    assert info['shares'] == 100 and info['cash'] == 0
    assert info['nav'] == 10000 and reward == 0
    assert truncated and not terminated
    assert len(env.history) == 4 and len(env.trades) == 3
    with pytest.raises(RuntimeError): env.step(-1)


def test_next_open_jump_does_not_give_new_holder_overnight_return():
    env = TradingEnv(bars((100,200,300), (100,220,300)), config())
    env.reset()
    _, reward, _, _, info = env.step(1)
    assert info['shares'] == 50
    assert info['nav'] == 11000
    assert reward == pytest.approx(np.log(1.1))
    assert env.trades[0]['price'] == 200
    assert env.step(-1)[4]['nav'] == 15000


def test_zero_action_is_half_weight():
    env = TradingEnv(bars(), config())
    env.reset()
    info = env.step(0)[4]
    assert info['shares'] == 50 and info['cash'] == 5000
    assert info['requested_weight'] == info['executed_weight'] == .5
    assert info['turnover'] == .5


def test_fees_once_minimum_commission_tax_and_slippage():
    env = TradingEnv(bars(), TradingConfig(commission=.001, slippage=.01,
        min_commission=5, sell_tax=.01, max_participation=1))
    env.reset()
    _, reward, _, _, info = env.step(1)
    # 98 shares * 101 + 9.898 commission = 9907.898.
    assert info['shares'] == 98
    assert info['cash'] == pytest.approx(92.102)
    assert info['cost'] == pytest.approx(107.898)
    assert reward == pytest.approx(np.log(9892.102 / 10000))
    info = env.step(-1)[4]
    # sale at 99, commission 9.702, tax 97.02, cash proceeds 9595.278.
    assert info['cash'] == pytest.approx(9687.380)
    assert info['shares'] == 0


def test_minimum_commission_blocks_unaffordable_single_lot():
    env = TradingEnv(bars(), config(initial_cash=104, min_commission=5))
    env.reset()
    assert env.step(1)[4]['shares'] == 0


def test_custom_lots_and_causal_participation_cap():
    env = TradingEnv(bars(volumes=[1000,1000000,1000000,1000000]),
        TradingConfig(commission=0, slippage=0, lot_size=10, max_participation=.025))
    env.reset()
    info = env.step(1)[4]
    assert info['shares'] == 20 and info['turnover'] == .2
    assert info['executed_weight'] == .2


def test_zero_volume_no_fill_but_time_advances():
    env = TradingEnv(bars(volumes=[100000,0,100000,100000]), config())
    env.reset()
    info = env.step(1)[4]
    assert info['nav'] == 10000 and info['shares'] == 0
    assert env.current_step == 1


def test_sell_next_day_is_unlocked_under_t_plus_one():
    env = TradingEnv(bars(), config(t_plus_one=True))
    env.reset()
    env.step(1)
    assert env.step(-1)[4]['shares'] == 0


def test_slice_end_truncates_without_liquidation_or_future_reward():
    env = TradingEnv(bars((100,100,110,1000)), config(), start=1, end=2)
    env.reset()
    _, _, terminated, truncated, info = env.step(1)
    assert truncated and not terminated and info['shares'] == 90
    assert info['nav'] == 10000
    assert len(env.trades) == 1
    assert info['date'] == '2024-01-03'


def test_random_episodes_reproducible_and_stay_in_bounds():
    env = TradingEnv(bars((100,)*20), config(), start=4, end=15,
        random_start=True, episode_length=3)
    obs, info = env.reset(seed=31)
    first = env.current_step
    assert 4 <= first <= 12 and obs.dtype == np.float32
    for _ in range(3): _, reward, terminated, truncated, _ = env.step(-1)
    assert reward == 0 and truncated and not terminated
    assert env.current_step == first + 3 <= 15
    env.reset(seed=31)
    assert env.current_step == first


@pytest.mark.parametrize('kwargs', [{'initial_cash':0}, {'commission':-1}, {'slippage':1},
    {'lot_size':0}, {'lot_size':1.5}, {'max_participation':2}, {'drawdown_penalty':-1}])
def test_invalid_configuration_rejected(kwargs):
    with pytest.raises(ValueError): TradingConfig(**kwargs)


def test_gym_and_sb3_environment_contracts():
    from gymnasium.utils.env_checker import check_env
    from stable_baselines3.common.env_checker import check_env as sb3_check
    env = TradingEnv(bars((100,)*20), config())
    check_env(env, skip_render_check=True)
    sb3_check(env)


def test_observations_are_finite_and_within_declared_bounds():
    env = TradingEnv(bars((1e-40,1e10,1e10)), config())
    obs, _ = env.reset()
    assert np.isfinite(env.observation_space.low).all()
    assert np.isfinite(env.observation_space.high).all()
    obs, *_ = env.step(-1)
    assert np.isfinite(obs).all() and env.observation_space.contains(obs)


def test_future_volume_changes_cannot_inflate_decision_capacity():
    frame = bars(volumes=[1000,5000,5000,5000])
    future = frame.copy()
    future.iloc[1:,4] = 10000000
    first, second = TradingEnv(frame),TradingEnv(future)
    obs1,_ = first.reset()
    obs2,_ = second.reset()
    np.testing.assert_array_equal(obs1,obs2)
    assert first.step(1)[4]['shares'] == second.step(1)[4]['shares'] == 10


def test_reward_penalty_uses_executed_turnover_and_incremental_drawdown():
    frame = bars((100,100,100),(100,90,90))
    env = TradingEnv(frame,config(turnover_penalty=.1,drawdown_penalty=2))
    env.reset()
    _, reward, *_ = env.step(0)
    # Half allocation loses 5%; turnover is .5, and new drawdown .05.
    assert reward == pytest.approx(np.log(.95)-.05-.1)


@pytest.mark.parametrize('action',[np.nan,np.inf,[0,1]])
def test_invalid_actions_rejected_without_advancing(action):
    env = TradingEnv(bars(),config())
    env.reset()
    with pytest.raises(ValueError): env.step(action)
    assert env.current_step == 0 and len(env.history) == 1
