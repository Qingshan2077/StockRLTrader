"""Calendar folds and deterministic preflight resource accounting."""
from bisect import bisect_left
import math
import pandas as pd
from stockrl.research.contracts import (
    BudgetPreview, FoldPlan, ResearchError, ResearchPreview, ResearchProtocolDraft,
    SessionWindow, TrainingBudget,
)


def rollout_steps(budget: TrainingBudget) -> int:
    if budget.algorithm == 'SAC':
        return 1
    rollout = max(2,min(128,budget.requested_timesteps // 4))
    return rollout - rollout % 2


def checkpoint_steps(budget: TrainingBudget, maximum: int = 20) -> tuple[int,...]:
    if type(maximum) is not int or not 1 <= maximum <= 20:
        raise ValueError('checkpoint budget must be between 1 and 20')
    rollout = rollout_steps(budget)
    updates = math.ceil(budget.requested_timesteps / rollout)
    count = min(maximum,updates)
    steps = tuple(sorted({math.ceil(i * updates / count) * rollout for i in range(1,count+1)}))
    if budget.algorithm == 'SAC':
        learning_starts = min(100,budget.requested_timesteps // 4)
        steps = tuple(s for s in steps if s > learning_starts)
    return steps


def budget_preview(assets: int, folds: int, seeds: int, budget: TrainingBudget,
                   maximum_checkpoints: int = 20) -> BudgetPreview:
    if any(type(n) is not int or n < 0 for n in (assets,folds,seeds)):
        raise ValueError('budget dimensions must be nonnegative integers')
    units = assets * folds * seeds
    requested = units * budget.requested_timesteps
    if units > 250 or requested > 25000000:
        raise ResearchError('RESEARCH_BUDGET_EXCEEDED','maximum 250 units and 25000000 requested steps')
    rollout = rollout_steps(budget)
    actual = math.ceil(budget.requested_timesteps / rollout) * rollout
    checkpoints = checkpoint_steps(budget, maximum_checkpoints)
    candidates = units * len(checkpoints)
    calibration = units * 21
    tests = units * 3 * 10  # RL plus all nine locked references, each scenario
    return BudgetPreview(unit_count=units, requested_total_steps=requested,
        rollout_upper_bound=units*actual, actual_steps_per_unit=actual, rollout_steps=rollout,
        checkpoint_steps=checkpoints, checkpoint_evaluations=candidates,
        calibration_evaluations=calibration,test_evaluations=tests,
        extra_evaluations=candidates+calibration+tests)


def _sessions(frame) -> list[str]:
    if isinstance(frame,pd.DataFrame):
        if 'session' not in frame: raise ValueError('calendar requires session')
        frame = frame.loc[frame['is_open'].eq(True)] if 'is_open' in frame else frame
        values = frame['session']
    else:
        values = frame
    result = [pd.Timestamp(s).date().isoformat() for s in values]
    if result != sorted(set(result)):
        raise ValueError('sessions must be unique and increasing')
    return result


def _build(draft: ResearchProtocolDraft, sessions, instrument_id=None):
    days = _sessions(sessions)
    first = pd.Timestamp(draft.first_test_session)
    if pd.isna(first): raise ValueError('first test boundary is invalid')
    folds, omitted = [], []
    for index in range(draft.fold_count):
        boundary = first + pd.DateOffset(months=6*index)
        dates = [boundary-pd.DateOffset(months=72),boundary-pd.DateOffset(months=12),
                 boundary,boundary+pd.DateOffset(months=6)]
        calendar = [d.date().isoformat() for d in dates]
        offsets = [bisect_left(days,d) for d in calendar]
        if offsets[0] == 0 or offsets[-1] >= len(days):
            omitted.append(f'fold_{index+1}: insufficient initial observation or complete test end boundary')
            continue
        windows = []
        for n in range(3):
            start,end = offsets[n:n+2]
            if start >= end:
                break
            windows.append(SessionWindow(calendar_start=calendar[n],calendar_end=calendar[n+1],
                start_session=days[start],end_session=days[end],initial_session=days[start-1],
                reward_sessions=tuple(days[start:end]),warmup_sessions=tuple(days[max(0,start-252):start])))
        if len(windows) != 3:
            omitted.append(f'fold_{index+1}: empty reward interval')
            continue
        folds.append(FoldPlan(fold_id=f'fold_{index+1}',instrument_id=instrument_id,
                              train=windows[0],validation=windows[1],test=windows[2]))
    return folds,omitted


def build_fold_plan(draft: ResearchProtocolDraft, sessions) -> list[FoldPlan]:
    """Resolve half-open rewards; the previous close is observation-only.

    A mapping of instrument IDs to calendars supports securities with distinct
    exchange sessions. A terminal boundary must be present to prove completeness.
    """
    if isinstance(sessions,dict):
        return [fold for asset in draft.instrument_ids for fold in _build(draft,sessions[asset],asset)[0]]
    return _build(draft,sessions)[0]


def preview_research(draft: ResearchProtocolDraft, sessions,
                     blockers: tuple[str,...] = ()) -> ResearchPreview:
    if isinstance(sessions,dict):
        built = [_build(draft,sessions[asset],asset) for asset in draft.instrument_ids]
        folds = [fold for plans,_ in built for fold in plans]
        omitted = [reason for _,reasons in built for reason in reasons]
        budget = budget_preview(1,len(folds),len(draft.seeds),draft.training_budget,draft.checkpoint_budget)
    else:
        folds,omitted = _build(draft,sessions)
        budget = budget_preview(len(draft.instrument_ids),len(folds),len(draft.seeds),draft.training_budget,draft.checkpoint_budget)
    if not folds: blockers = (*blockers,'NO_COMPLETE_FOLDS')
    return ResearchPreview(fold_plan=tuple(folds),budget=budget,checkpoint_steps=budget.checkpoint_steps,
        qualification=draft.qualification,blockers=blockers,omitted_folds=tuple(omitted))
