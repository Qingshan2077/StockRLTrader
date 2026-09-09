import numpy as np
import pandas as pd
import pytest

from stockrl.research.baselines import reference_weight, REFERENCE_IDS, baseline_cache_key


def test_weekly_means_no_order_between_known_week_boundaries():
    assert reference_weight('weekly_50', [100], '2024-01-08', '2024-01-05') == .5
    assert reference_weight('weekly_50', [100], '2024-01-09', '2024-01-08') is None
    assert reference_weight('weekly_50', [100], '2024-01-02', '2023-12-29') == .5


def test_transparent_fixed_trend_and_sample_volatility_rules():
    assert len(REFERENCE_IDS) == 8
    assert reference_weight('buy_hold', [100], '2024-01-03', '2024-01-02') == 1
    assert reference_weight('fixed_25', [100], '2024-01-03', '2024-01-02') == .25
    assert reference_weight('trend_20', np.arange(1, 21), '2024-01-03', '2024-01-02') == 1
    assert reference_weight('trend_20', np.ones(20), '2024-01-03', '2024-01-02') == 0
    returns = np.array([.03, -.02] * 10)
    closes = np.r_[100, 100*np.cumprod(1+returns)]
    assert reference_weight('vol_target_10', closes[:-1], '', '') == 0
    assert reference_weight('vol_target_10', closes, '', '') == pytest.approx(.1/(np.std(returns, ddof=1)*np.sqrt(252)))


def test_cache_full_inputs_and_fixed_seed_independence():
    args = dict(dataset_fingerprint='abc', market_profile={'fee': .1}, window={'sessions':['a']}, strategy_id='fixed_50', initial_cash=10000)
    assert baseline_cache_key(**args, seed=42) == baseline_cache_key(**args, seed=43)
    assert baseline_cache_key(**args) != baseline_cache_key(**{**args, 'initial_cash': 20000})
    assert baseline_cache_key(**args) != baseline_cache_key(**{**args, 'market_profile': {'fee': .2}})
    assert baseline_cache_key(**{**args, 'strategy_id':'matched_fixed'}, seed=42, matched_weight=.5) != baseline_cache_key(**{**args, 'strategy_id':'matched_fixed'}, seed=43, matched_weight=.5)


def test_references_replay_real_account_with_cash_and_weekly_no_order():
    from test_cost_scenarios import research_fixture
    from stockrl.research.runner import evaluate_strategy
    bundle,protocol,key=research_fixture()
    window=protocol.fold_plan[0].test
    cash=evaluate_strategy(bundle,window,None,strategy_id='cash')
    assert cash['metrics']['net_return']==0 and not cash['trades']
    buy=evaluate_strategy(bundle,window,None,strategy_id='buy_hold')
    assert buy['trades'] and all(t['side']=='buy' for t in buy['trades'])
    weekly=evaluate_strategy(bundle,window,None,strategy_id='weekly_50')
    for row in weekly['decisions']:
        if pd.Timestamp(row['session']).isocalendar()[:2] == pd.Timestamp(row['decision_session']).isocalendar()[:2]:
            assert row['requested_weight'] is None and row['filled_qty']==0


def test_replay_exposes_at_most_252_past_sessions():
    from stockrl.research.runner import window_bundle
    from test_cost_scenarios import research_fixture
    bundle,protocol,key=research_fixture()
    window=protocol.fold_plan[0].test.model_copy(update={'warmup_sessions':()})
    limited=window_bundle(bundle,window)
    assert limited.sessions[0].session==window.initial_session
    assert limited.sessions[-1].session==window.reward_sessions[-1]


def test_repeated_references_reuse_preparation_without_sharing_mutable_frames(monkeypatch):
    from stockrl.research.runner import window_bundle
    from stockrl.env_v2 import TradingEnvV2
    import stockrl.env_v2 as module
    from test_cost_scenarios import research_fixture
    bundle,protocol,key=research_fixture()
    window=protocol.fold_plan[0].test
    limited=window_bundle(bundle,window)
    assert window_bundle(bundle,window) is limited
    first=TradingEnvV2(limited,window)
    expected=first.features.copy()
    first.features.iloc[0,0]=999
    def forbidden(*args,**kwargs): raise AssertionError('repeated immutable input rebuilt features')
    monkeypatch.setattr(module,'build_features',forbidden)
    second=TradingEnvV2(limited,window)
    pd.testing.assert_frame_equal(second.features,expected)
