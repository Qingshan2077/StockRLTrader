import numpy as np
from tests.v2.test_corporate_actions import bundle


def test_gym_next_open_no_order_and_episode_boundary():
    from stockrl.env_v2 import TradingEnvV2
    env=TradingEnvV2(bundle((100,50,60)),('2024-01-03','2024-01-04'),initial_cash=1000)
    obs,info=env.reset(seed=42)
    assert obs.shape==(12,) and info['nav']==1000
    obs,reward,terminated,truncated,info=env.step(np.array([1]))
    assert env.state.shares==20 and not terminated and not truncated
    obs,reward,terminated,truncated,info=env.step(None)
    assert env.state.shares==20 and info['nav']==1200 and truncated
    assert len(env.trades)==1 and len(env.decision_logs)==2

def test_buy_hold_does_not_sell_when_receivables_increase_target_nav():
    from stockrl.env_v2 import TradingEnvV2
    from stockrl.market.contracts import AccountState
    from tests.v2.test_corporate_actions import dividend
    env=TradingEnvV2(bundle(actions=[dividend()]),('2024-01-03','2024-01-04'),initial_cash=1000)
    env.reset()
    env.state=AccountState(0,10,10,{},1000,1000)
    _,reward,_,_,info=env.step([1])
    assert env.state.shares==10 and env.state.receivables=={'d':20} and reward==0
    assert len(env.trades)==0 and info['receivable_fraction']==.02


def test_gym_checker_v2_and_legacy_observation_unchanged():
    from gymnasium.utils.env_checker import check_env
    from stockrl.env_v2 import TradingEnvV2
    from stockrl.env import TradingEnv
    from stockrl.data import make_demo_data
    check_env(TradingEnvV2(bundle((100,100,100)),('2024-01-03','2024-01-04')),skip_render_check=True)
    obs,_=TradingEnv(make_demo_data(5)).reset()
    assert obs.shape==(11,)
