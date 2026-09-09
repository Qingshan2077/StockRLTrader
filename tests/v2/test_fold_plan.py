import pandas as pd
from stockrl.research.contracts import ResearchProtocolDraft
from stockrl.research.folds import build_fold_plan, preview_research


def draft(**kw):
    return ResearchProtocolDraft(protocol_id='p', instrument_ids=('A',), dataset_ids=('d',), dataset_fingerprints={'d':'a'*64}, market_profile_ids=('m',), market_profile_fingerprints={'m':'b'*64}, asset_selection_note='preselected', first_test_session='2020-01-31', **kw)


def test_calendar_half_open_warmup_and_tail():
    sessions = pd.DataFrame({'session':pd.bdate_range('2013-01-01','2021-01-31').strftime('%Y-%m-%d')})
    folds = build_fold_plan(draft(), sessions)
    assert len(folds) == 1  # second end boundary unavailable: never guess completeness
    fold = folds[0]
    assert fold.test.calendar_end == '2020-07-31'
    assert fold.test.reward_sessions[0] == '2020-01-31'
    assert fold.test.reward_sessions[-1] == '2020-07-30'
    assert fold.test.initial_session == '2020-01-30'
    assert len(fold.test.warmup_sessions) == 252
    assert set(fold.train_rewards).isdisjoint(fold.validation_rewards)
    assert set(fold.validation_rewards).isdisjoint(fold.test_rewards)
    assert preview_research(draft(), sessions).omitted_folds


def test_seeds_do_not_change_calendar_and_closed_sessions_excluded():
    sessions = pd.DataFrame({'session':pd.date_range('2013-01-01','2023-01-01').strftime('%Y-%m-%d')})
    sessions['is_open'] = pd.to_datetime(sessions.session).dt.dayofweek < 5
    a = build_fold_plan(draft(seeds=(1,)), sessions)
    b = build_fold_plan(draft(), sessions)
    assert a == b
    assert all(pd.Timestamp(s).dayofweek < 5 for f in a for s in f.test_rewards)
