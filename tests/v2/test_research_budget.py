import pytest
from stockrl.research.contracts import TrainingBudget, ResearchProtocol, ResearchError
from stockrl.research.folds import budget_preview, checkpoint_steps
from test_fold_plan import draft


def test_rollout_alignment_postupdate_and_caps():
    budget = TrainingBudget(requested_timesteps=100001)
    steps = checkpoint_steps(budget)
    assert len(steps) == 20 and steps == tuple(sorted(set(steps)))
    assert all(s > 0 and s % 128 == 0 for s in steps)
    assert steps[-1] == 100096
    preview = budget_preview(2,4,5,budget)
    assert preview.requested_total_steps == 4000040
    assert preview.rollout_upper_bound == 4003840
    with pytest.raises(ResearchError, match='RESEARCH_BUDGET_EXCEEDED'):
        budget_preview(13,4,5,budget)
    with pytest.raises(ResearchError, match='RESEARCH_BUDGET_EXCEEDED'):
        budget_preview(1,6,5,TrainingBudget(requested_timesteps=1000000))


def test_tiny_budget_and_strict_validation():
    assert checkpoint_steps(TrainingBudget(requested_timesteps=1)) == (2,)
    with pytest.raises(ValueError): TrainingBudget(requested_timesteps=0)
    with pytest.raises(ValueError): TrainingBudget(requested_timesteps=1000001)
    with pytest.raises(ValueError): TrainingBudget(requested_timesteps=1.5)
    with pytest.raises(ValueError): draft(seeds=(42,42))


def test_protocol_canonical_lock_detects_nested_mutation():
    protocol = ResearchProtocol(**draft().model_dump(), fold_plan=(), checkpoint_steps=checkpoint_steps(draft().training_budget))
    lock = protocol.lock()
    assert lock.verify().protocol_id == 'p'
    changed = protocol.model_copy(update={'hypothesis':'different'})
    assert changed.lock().sha256 != lock.sha256
    assert protocol.lock().canonical_json == lock.canonical_json
    with pytest.raises(ValueError): lock.model_copy(update={'sha256':'0'*64}).verify()

def test_protocol_rejects_untrained_or_unaligned_checkpoints():
    import pandas as pd
    from stockrl.research.folds import build_fold_plan
    d = draft()
    folds = build_fold_plan(d,pd.DataFrame({'session':pd.bdate_range('2013-01-01','2023-01-01')}))
    with pytest.raises(ValueError):
        ResearchProtocol(**d.model_dump(),fold_plan=folds,checkpoint_steps=(1,))


def test_dates_and_seed_ranges_are_validated_at_boundary():
    with pytest.raises(ValueError): type(draft()).model_validate({**draft().model_dump(), 'first_test_session':'not a date'})
    with pytest.raises(ValueError): draft(seeds=(-1,))
