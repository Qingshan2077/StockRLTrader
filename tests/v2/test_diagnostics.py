import pytest
from stockrl.research.diagnostics import summarize_diagnostics


def test_boundary_includes_cash_and_separates_clipping():
    rows=[{'clipped_action':1,'raw_action':1.1,'requested_weight':1}]*506 + [{'clipped_action':-1,'raw_action':-1,'requested_weight':0}]*23 + [{'clipped_action':0,'raw_action':0,'requested_weight':.5}]*3
    r=summarize_diagnostics(rows)
    assert r.boundary_action_ratio==pytest.approx(529/532)
    assert r.clipping_ratio==pytest.approx(506/532)
    assert 'boundary_action_dominance' in r.warnings


def test_near_half_and_missing_raw_are_not_invented():
    r=summarize_diagnostics([{'requested_weight':.5}]*9783+[{'requested_weight':1}]*217)
    assert r.near_half_ratio==.9783
    assert r.clipping_ratio is None
    assert r.boundary_action_ratio is None
    assert 'near_half_dominance' in r.warnings
    assert 'raw_action' in r.unavailable
    assert 'training_log' in r.unavailable


def test_strict_ninety_percent_and_inclusive_half_thresholds():
    r=summarize_diagnostics([{'clipped_action':.999,'requested_weight':.45}]*9+[{'clipped_action':0,'requested_weight':.56}])
    assert 'boundary_action_dominance' not in r.warnings
    assert 'near_half_dominance' not in r.warnings


def test_drift_uses_preclip_values_and_only_eight_market_features():
    import numpy as np
    train=np.tile(np.arange(100).reshape(-1,1),(1,8))
    evaluation=np.ones((20,8))*50
    standardized=np.zeros((20,8)); standardized[:2,0]=11
    r=summarize_diagnostics([],training_features=train,evaluation_features=evaluation,standardized_features=standardized)
    assert r.feature_stats['0']['clipping_ratio']==.1
    assert r.feature_stats['0']['outside_training_quantiles_ratio']==0
    assert 'distribution_shift' in r.warnings
    with pytest.raises(ValueError):
        summarize_diagnostics([],training_features=np.zeros((2,12)),evaluation_features=np.zeros((2,12)),standardized_features=np.zeros((2,12)))


def test_regimes_use_decision_state_and_low_coverage_no_annualization():
    r=summarize_diagnostics([{'requested_weight':.5,'executed_weight':.25,'filled_qty':0,'fees':2}], regimes=[{'momentum_20':.1,'volatility_20':.2,'training_volatility_median':.1,'log_return_difference':.03,'position_weight_close':.25,'fees':2}])
    group=r.regimes['positive_high']
    assert group['sample_count']==1
    assert group['log_return_difference_sum']==.03
    assert group['low_coverage'] is True
    assert group['cost']==2
    assert r.summary['zero_fill_count']==1
    assert r.summary['target_actual_difference_count']==1


def test_missing_rollout_fields_are_null():
    r=summarize_diagnostics([],training_log=[{'actual_steps':128,'value_loss':.2}])
    assert r.training_log[0]['approx_kl'] is None
    assert r.training_log[0]['actual_steps']==128
