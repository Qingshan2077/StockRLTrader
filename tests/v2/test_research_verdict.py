import pytest
import pandas as pd
from stockrl.research.contracts import ResearchProtocol, ResearchProtocolDraft, UnitResult, UnitKey, CostResult, StrategyMetrics
from stockrl.research.folds import build_fold_plan, checkpoint_steps
from stockrl.research.report import summarize_research


def fixture(assets=('A',), folds=4, qualification='declared_holdout'):
    draft = ResearchProtocolDraft(protocol_id='p', instrument_ids=assets, dataset_ids=('d',), dataset_fingerprints={'d':'a'*64}, market_profile_ids=('m',), market_profile_fingerprints={'m':'b'*64}, asset_selection_note='locked', first_test_session='2020-01-01', fold_count=folds, qualification=qualification, scope='single_asset' if len(assets)==1 else 'asset_set')
    protocol = ResearchProtocol(**draft.model_dump(), fold_plan=build_fold_plan(draft, pd.bdate_range('2013-01-01','2023-01-01')), checkpoint_steps=checkpoint_steps(draft.training_budget))
    units = []
    for asset in assets:
        for fold in protocol.fold_plan:
            for seed in protocol.seeds:
                costs=[]
                for scenario in protocol.cost_scenarios:
                    reference=StrategyMetrics(cagr=.10, annualized_volatility=.1, max_drawdown=.1)
                    rl=reference.model_copy(update={'cagr': .11 if scenario=='base' else (.10 if scenario=='execution_x2' else -.5)})
                    costs.append(CostResult(scenario_id=scenario,model_sha256='a'*64, metrics={'rl':rl,'fixed_50':reference,'matched_fixed':reference}))
                units.append(UnitResult(key=UnitKey(instrument_id=asset,fold_id=fold.fold_id,seed=seed),cost_results=costs))
    return protocol,units


def change(units, scenario, strategy, **metrics):
    return [u.model_copy(update={'cost_results':tuple(c.model_copy(update={'metrics':{**c.metrics,strategy:c.metrics[strategy].model_copy(update=metrics)}}) if c.scenario_id==scenario else c for c in u.cost_results)}) for u in units]


def test_positive_fixture_x3_is_report_only_and_seed_medians():
    p,u=fixture()
    r=summarize_research(p,u)
    assert r.economic_outcome=='candidate_edge'
    assert r.assets['A']['base']['fixed_50']['cagr_difference']==pytest.approx(.01)
    assert r.assets['A']['winning_fold_fraction']==1
    assert r.assets['A']['absolute_cagr']==pytest.approx(.11)


@pytest.mark.parametrize('kwargs', [{'qualification':'exploratory'}, {'folds':3}])
def test_asset_verdict_cannot_bypass_parent_evidence_gates(kwargs):
    protocol, units = fixture(**kwargs)
    report = summarize_research(protocol, units)
    assert report.assets['A']['economic_outcome'] == 'insufficient_evidence'
    assert report.assets['A']['provisional_outcome'] == 'candidate_edge'


@pytest.mark.parametrize('mode', ['missing','failed','cancelled','fingerprint','coverage','duplicate','missing_cost','missing_reference'])
def test_missing_or_invalid_unit_never_passes(mode):
    p,u=fixture()
    if mode=='missing':u=u[:-1]
    elif mode=='duplicate':u=[*u,u[0]]
    elif mode=='missing_cost':u[0]=u[0].model_copy(update={'cost_results':u[0].cost_results[:1]})
    elif mode=='missing_reference':u[0]=u[0].model_copy(update={'cost_results':(u[0].cost_results[0].model_copy(update={'metrics':{'rl':StrategyMetrics()}}),*u[0].cost_results[1:])})
    else:u[0]=u[0].model_copy(update={ {'failed':'status','cancelled':'status','fingerprint':'fingerprints_valid','coverage':'rule_coverage_complete'}[mode]:mode if mode in ('failed','cancelled') else False})
    assert summarize_research(p,u).economic_outcome=='insufficient_evidence'


@pytest.mark.parametrize('scenario,metrics', [('base',{'cagr':.1}),('base',{'max_drawdown':.121}),('execution_x2',{'cagr':.099})])
def test_economic_failure(scenario,metrics):
    p,u=fixture()
    assert summarize_research(p,change(u,scenario,'rl',**metrics)).economic_outcome=='no_added_value'


def test_risk_denominator_and_exact_eighty_percent():
    p,u=fixture()
    u[:4]=change(u[:4],'base','rl',annualized_volatility=.5)
    assert summarize_research(p,u).economic_outcome=='candidate_edge'
    u[4:5]=change(u[4:5],'base','rl',annualized_volatility=.5)
    assert summarize_research(p,u).economic_outcome=='insufficient_evidence'


def test_short_exploratory_and_asset_set_coverage():
    p,u=fixture(folds=3)
    assert summarize_research(p,u).economic_outcome=='insufficient_evidence'
    p,u=fixture(qualification='exploratory')
    r=summarize_research(p,u)
    assert r.economic_outcome=='insufficient_evidence'
    assert r.provisional_outcome=='candidate_edge'
    p,u=fixture(assets=('A','B','C','D'))
    assert summarize_research(p,u).economic_outcome=='insufficient_evidence'
    p,u=fixture(assets=('A','B','C','D','E'))
    assert summarize_research(p,u).economic_outcome=='candidate_edge'


def test_folds_not_pooled_seeds_and_both_references_must_win():
    p,u=fixture()
    # Three good seeds and two catastrophic seeds per fold: median stays positive.
    u=[change([x],'base','rl',cagr=-10)[0] if x.key.seed in (45,46) else x for x in u]
    assert summarize_research(p,u).economic_outcome=='candidate_edge'
    u=change(u,'base','matched_fixed',cagr=.2)
    assert summarize_research(p,u).economic_outcome=='no_added_value'


def test_metrics_count_reward_sessions_and_cash_has_null_sharpe():
    from stockrl.research.report import calculate_metrics
    rows=[{'nav':100,'weight':1},{'nav':100,'weight':0,'fees':1,'receivables':3},{'nav':100,'weight':0,'fees':2,'receivables':3}]
    r=calculate_metrics(rows,[])
    assert r.reward_sessions==2
    assert r.cagr==0
    assert r.sharpe is None
    assert r.average_exposure==0
    assert r.fees==3
    assert r.final_receivables==3
    assert calculate_metrics(rows[:2],[]).annualized_volatility is None


def test_metrics_drawdown_uses_initial_nav_and_actual_session_annualization():
    from stockrl.research.report import calculate_metrics
    r=calculate_metrics([{'nav':100},{'nav':90},{'nav':99}],[])
    assert r.max_drawdown==pytest.approx(.1)
    assert r.net_return==pytest.approx(-.01)
    assert r.cagr==pytest.approx(.99**126-1)


def test_overlapping_fold_and_wrong_seed_block_qualification():
    p,u=fixture()
    folds=list(p.fold_plan)
    folds[-1]=folds[-1].model_copy(update={'test':folds[0].test})
    assert summarize_research(p.model_copy(update={'fold_plan':tuple(folds)}),u).economic_outcome=='insufficient_evidence'
    assert summarize_research(p.model_copy(update={'seeds':(42,43,44,45,47)}),u).economic_outcome=='insufficient_evidence'


def test_cash_and_two_of_four_winning_folds_fail():
    p,u=fixture()
    cash=change(change(u,'base','rl',cagr=0),'base','fixed_50',cagr=0)
    cash=change(cash,'base','matched_fixed',cagr=0)
    assert summarize_research(p,cash).economic_outcome=='no_added_value'
    u=[change([x],'base','rl',cagr=.099)[0] if x.key.fold_id in ('fold_3','fold_4') else x for x in u]
    assert summarize_research(p,u).economic_outcome=='no_added_value'


def test_risk_comparability_is_visible_for_every_seed():
    p,u=fixture()
    u[:1]=change(u[:1],'base','rl',annualized_volatility=.5)
    r=summarize_research(p,u)
    assert r.assets['A']['folds']['fold_1']['risk_comparability']['42']=='risk_mismatch'
    assert r.assets['A']['folds']['fold_1']['risk_comparability']['43']=='comparable'
