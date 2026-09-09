from dataclasses import replace
import pytest
from stockrl.market.contracts import MarketProfile, MarketFeeInterval
from stockrl.research.runner import cost_profile


def test_cost_stress_changes_only_execution_fees_in_all_dated_intervals():
    fields=dict(effective_start='2024-01-01',effective_end='2024-12-31',commission_rate=.001,commission_min=5,sell_tax_rate=.002,other_fee_rate=.0001,dividend_tax_rate=.1,slippage=.001,participation_cap=.1,rule_sources=('synthetic',))
    profile=MarketProfile(profile_id='test',version='1',exchange='SSE',security_type='common_stock',currency='CNY',buy_lot=100,sell_odd_lot=True,t_plus_one=True,fee_intervals=(MarketFeeInterval(**fields),),**fields)
    stress=cost_profile(profile,'execution_x3')
    assert stress.for_session('2024-01-02').commission_rate == .003
    assert stress.for_session('2024-01-02').commission_min == 15
    assert stress.for_session('2024-01-02').slippage == .003
    assert stress.sell_tax_rate==profile.sell_tax_rate and stress.other_fee_rate==profile.other_fee_rate
    assert profile.commission_min == 5
    with pytest.raises(ValueError): cost_profile(replace(profile,slippage=.5),'execution_x2')


def research_fixture():
    import pandas as pd
    from stockrl.market.contracts import Bar,MarketSession,Tradability,MarketDatasetV2
    from stockrl.research.contracts import SessionWindow,FoldPlan,ResearchProtocol,TrainingBudget,UnitKey
    from stockrl.research.folds import checkpoint_steps
    dates=[d.date().isoformat() for d in pd.bdate_range('2024-01-01',periods=65)]
    prices=[100 + i*.1 + (i%5)*.4 for i in range(len(dates))]
    profile=MarketProfile(profile_id='synthetic',version='1',exchange='SSE',security_type='common_stock',currency='CNY',effective_start=dates[0],effective_end=dates[-1],buy_lot=1,sell_odd_lot=True,t_plus_one=True,commission_rate=.001,commission_min=.2,sell_tax_rate=.001,other_fee_rate=.0001,dividend_tax_rate=0,slippage=.001,participation_cap=.1,rule_sources=('synthetic',))
    bundle=MarketDatasetV2(metadata={'instrument_id':'SSE:600000'},bars=tuple(Bar(d,p,p+1,p-1,p,100000) for d,p in zip(dates,prices)),sessions=tuple(MarketSession(d,d+'T09:30:00+08:00',d+'T15:00:00+08:00',True) for d in dates),actions=(),tradability=tuple(Tradability(d,d+'T09:00:00+08:00',True,True,1000,1,p,'normal') for d,p in zip(dates,prices)),market_profile=profile,file_hashes={},fingerprint='a'*64)
    def window(a,b):
        return SessionWindow(calendar_start=dates[a],calendar_end=dates[b],start_session=dates[a],end_session=dates[b],initial_session=dates[a-1],reward_sessions=tuple(dates[a:b]),warmup_sessions=tuple(dates[max(0,a-252):a]))
    budget=TrainingBudget(requested_timesteps=8,episode_length=8)
    protocol=ResearchProtocol(protocol_id='smoke',instrument_ids=('SSE:600000',),dataset_ids=('d',),dataset_fingerprints={'d':'a'*64},market_profile_ids=('synthetic',),market_profile_fingerprints={'synthetic':'b'*64},asset_selection_note='synthetic test',first_test_session=dates[50],fold_count=1,seeds=(42,),training_budget=budget,checkpoint_budget=2,checkpoint_steps=checkpoint_steps(budget,2),fold_plan=(FoldPlan(fold_id='f1',train=window(25,40),validation=window(40,50),test=window(50,64)),))
    return bundle,protocol,UnitKey(instrument_id='SSE:600000',fold_id='f1',seed=42)


def test_window_suspension_keeps_causal_valuation_anchor():
    from stockrl.research.runner import window_bundle
    from stockrl.market.features import build_features
    bundle, protocol, _ = research_fixture()
    window = protocol.fold_plan[0].validation
    days = [s.session for s in bundle.sessions]
    window = window.model_copy(update={'warmup_sessions': tuple(days[10:40])})
    suspended = replace(bundle, bars=tuple(bar for bar in bundle.bars if bar.session != days[10]),
        tradability=tuple(replace(row, can_buy_open=False, can_sell_open=False, reason='suspended')
                          if row.session == days[10] else row for row in bundle.tradability))
    bounded = window_bundle(suspended, window)
    assert bounded.bars[0].session == days[9]
    assert len(window.warmup_sessions) == 30
    result = build_features(bounded)
    assert result.loc[window.reward_sessions[0]].notna().all()


@pytest.mark.parametrize('algorithm',['PPO','SAC'])
def test_unit_replays_same_selected_model_and_all_references(tmp_path,algorithm):
    import json
    from stockrl.research.runner import run_unit
    bundle,protocol,key=research_fixture()
    from stockrl.research.folds import checkpoint_steps
    budget=protocol.training_budget.model_copy(update={'algorithm':algorithm})
    protocol=protocol.model_copy(update={'training_budget':budget,'checkpoint_steps':checkpoint_steps(budget,2)})
    result=run_unit(protocol,key,tmp_path/'unit',None,bundle=bundle)
    assert result.status=='completed' and len(result.cost_results)==3
    assert len({c.model_sha256 for c in result.cost_results})==1
    assert all(len(c.metrics)==10 for c in result.cost_results)
    assert len(result.candidates)==2 and all(c.gradient_updates>0 for c in result.candidates)
    assert (tmp_path/'unit/model.zip').exists()
    assert json.loads((tmp_path/'unit/diagnostics.json').read_text())
    decisions=json.loads((tmp_path/'unit/base/rl/decisions.json').read_text())
    assert all(r['seed']==42 and r['checkpoint_id']==result.selected_checkpoint_id for r in decisions)
    if algorithm=='PPO': assert all(r['policy_std']>0 for r in decisions)
    references=json.loads((tmp_path/'unit/base/fixed_50/decisions.json').read_text())
    assert all(r['seed']==42 and r['fold_id']=='f1' for r in references)


def test_future_test_changes_cannot_change_normalizer_or_validation_selection(tmp_path):
    from stockrl.research.runner import run_unit
    bundle,protocol,key=research_fixture()
    first=run_unit(protocol,key,tmp_path/'first',None,bundle=bundle)
    boundary=protocol.fold_plan[0].test.start_session
    altered=replace(bundle,bars=tuple(replace(b,open=b.open*1.1,high=b.high*1.1,low=b.low*1.1,close=b.close*1.1) if b.session>=boundary else b for b in bundle.bars),fingerprint='c'*64)
    second=run_unit(protocol,key,tmp_path/'second',None,bundle=altered)
    assert first.selected_checkpoint_id==second.selected_checkpoint_id
    assert first.matched_weight==second.matched_weight
    assert [c.validation_log_return for c in first.candidates]==[c.validation_log_return for c in second.candidates]
    assert (tmp_path/'first/normalizer.json').read_bytes()==(tmp_path/'second/normalizer.json').read_bytes()
