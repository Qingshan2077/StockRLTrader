"""Paired, preregistered research verdicts and shared v2 account metrics."""
from __future__ import annotations

import math
from statistics import median

import numpy as np

from stockrl.research.contracts import ResearchReport, StrategyMetrics


def calculate_metrics(history, trades):
    """History includes an initial NAV row; subsequent rows are reward sessions."""
    rows = history.to_dict('records') if hasattr(history, 'to_dict') else list(history)
    nav = np.asarray([r['nav'] for r in rows], dtype=float)
    if len(nav) < 2 or not np.isfinite(nav).all() or (nav <= 0).any():
        raise ValueError('metrics require initial and positive finite reward-session NAV')
    returns = nav[1:] / nav[:-1] - 1
    n = len(returns)
    vol = float(np.std(returns, ddof=1) * np.sqrt(252)) if n > 1 else None
    exponent = math.log(nav[-1] / nav[0]) * 252 / n
    cagr = math.expm1(exponent) if exponent < math.log(np.finfo(float).max) else None
    return StrategyMetrics(net_return=float(nav[-1]/nav[0]-1), cagr=cagr,
        annualized_volatility=vol, sharpe=float(returns.mean()*252/vol) if vol and vol > 1e-12 else None,
        max_drawdown=float(np.max(1-nav/np.maximum.accumulate(nav))),
        fees=sum(r.get('fees', r.get('cost', 0)) for r in rows[1:]),
        turnover=sum(r.get('turnover', 0) for r in rows[1:]),
        average_exposure=float(np.mean([r.get('position_weight_close', r.get('weight', 0)) for r in rows[1:]])),
        trade_count=len(trades), reward_sessions=n, final_nav=float(nav[-1]),
        final_receivables=rows[-1].get('receivables', 0))


def exposure_qualification(protocol, records):
    """Instrument IDs must be canonical; exposure endpoints are inclusive."""
    if protocol.qualification == 'exploratory':
        return 'exploratory'
    for asset in protocol.instrument_ids:
        for fold in protocol.fold_plan:
            if fold.instrument_id not in (None, asset):
                continue
            for record in records:
                if record.instrument_id == asset and any(
                    record.start_session <= session <= record.end_session
                    for session in fold.test.reward_sessions
                ):
                    return 'exploratory'
    return 'declared_holdout'


def _key(key):
    return key.instrument_id, key.fold_id, key.seed


def summarize_research(protocol, units):
    """Keep every planned unit in the denominator and seed-pair before folds."""
    units = tuple(units)
    reasons = []
    plans = {a: [f for f in protocol.fold_plan if f.instrument_id in (None, a)] for a in protocol.instrument_ids}
    expected = {(a, f.fold_id, s) for a, folds in plans.items() for f in folds for s in protocol.seeds}
    indexed = {_key(u.key): u for u in units}
    complete = len(indexed) == len(units) and set(indexed) == expected and bool(expected)
    if not complete:
        reasons.append('planned_units_incomplete_or_duplicate')
    if any(u.status != 'completed' for u in units):
        reasons.append('unit_not_completed')
    if any(not u.fingerprints_valid for u in units):
        reasons.append('invalid_fingerprint')
    if any(not u.rule_coverage_complete for u in units):
        reasons.append('rule_coverage_incomplete')
    if tuple(protocol.seeds) != (42, 43, 44, 45, 46):
        reasons.append('requires_five_fixed_seeds')
    for asset, folds in plans.items():
        sessions = [s for f in folds for s in f.test.reward_sessions]
        if len(folds) < 4 or len({f.fold_id for f in folds}) != len(folds) or any(not f.test.reward_sessions for f in folds) or len(sessions) != len(set(sessions)):
            reasons.append(f'{asset}:requires_four_complete_nonoverlapping_folds')
    if protocol.scope == 'asset_set' and len(protocol.instrument_ids) < 5:
        reasons.append('requires_five_assets')
    valid_metrics = True
    for u in units:
        costs = {c.scenario_id: c for c in u.cost_results}
        if len(costs) != 3 or len(u.cost_results) != 3:
            valid_metrics = False
        for c in costs.values():
            for name in ('rl', 'fixed_50', 'matched_fixed'):
                m = c.metrics.get(name)
                if m is None or any(getattr(m, k) is None for k in ('cagr', 'annualized_volatility', 'max_drawdown')):
                    valid_metrics = False
    if not valid_metrics:
        reasons.append('missing_scenario_or_metrics')
    technical = 'completed'
    if any(u.status == 'failed' for u in units):
        technical = 'failed'
    elif any(u.status == 'cancelled' for u in units):
        technical = 'cancelled'
    elif not complete or any(u.status != 'completed' or not u.fingerprints_valid or not u.rule_coverage_complete for u in units) or not valid_metrics:
        technical = 'incomplete'
    assets = {}
    if complete and valid_metrics:
        for asset, folds in plans.items():
            evidence = {'folds': {}, 'risk_comparable_fraction': 0}
            comparable = 0
            for f in folds:
                paired = {}
                for scenario in protocol.cost_scenarios:
                    samples = [next(c for c in indexed[(asset, f.fold_id, s)].cost_results if c.scenario_id == scenario).metrics for s in protocol.seeds]
                    paired[scenario] = {r: {'cagr_difference': median(m['rl'].cagr-m[r].cagr for m in samples),
                        'drawdown_difference': median(m['rl'].max_drawdown-m[r].max_drawdown for m in samples)} for r in protocol.primary_reference_ids}
                    if scenario == 'base':
                        paired['absolute_cagr'] = median(m['rl'].cagr for m in samples)
                        paired['risk_comparability'] = {}
                        for seed, m in zip(protocol.seeds, samples):
                            v, ref = m['rl'].annualized_volatility, m['matched_fixed'].annualized_volatility
                            is_comparable = abs(v-ref) <= max(.02, .2*ref) + 1e-12
                            comparable += is_comparable
                            paired['risk_comparability'][str(seed)] = 'comparable' if is_comparable else 'risk_mismatch'
                evidence['folds'][f.fold_id] = paired
            if not folds:
                continue
            evidence['risk_comparable_fraction'] = comparable/(len(folds)*len(protocol.seeds))
            if evidence['risk_comparable_fraction'] < .8:
                reasons.append(f'{asset}:risk_mismatch')
            for scenario in protocol.cost_scenarios:
                evidence[scenario] = {r: {k: median(f[scenario][r][k] for f in evidence['folds'].values()) for k in ('cagr_difference','drawdown_difference')} for r in protocol.primary_reference_ids}
            evidence['absolute_cagr'] = median(f['absolute_cagr'] for f in evidence['folds'].values())
            evidence['winning_fold_fraction'] = sum(all(f['base'][r]['cagr_difference'] > 0 for r in protocol.primary_reference_ids) for f in evidence['folds'].values()) / len(folds)
            gates = {
                'positive_base': all(evidence['base'][r]['cagr_difference'] > 0 for r in protocol.primary_reference_ids),
                'winning_folds': evidence['winning_fold_fraction'] >= .6,
                'drawdown': all(evidence['base'][r]['drawdown_difference'] <= .02 + 1e-12 for r in protocol.primary_reference_ids),
                'execution_x2': all(evidence['execution_x2'][r]['cagr_difference'] >= 0 for r in protocol.primary_reference_ids),
            }
            evidence['gates'] = gates
            evidence['economic_outcome'] = 'candidate_edge' if all(gates.values()) else 'no_added_value'
            assets[asset] = evidence
    outcome = 'insufficient_evidence'
    if not reasons:
        winning = sum(a['economic_outcome'] == 'candidate_edge' for a in assets.values()) / len(assets)
        group_positive = all(median(a['base'][r]['cagr_difference'] for a in assets.values()) > 0 for r in protocol.primary_reference_ids)
        outcome = 'candidate_edge' if winning >= .6 and group_positive else 'no_added_value'
    provisional = None
    if protocol.qualification == 'exploratory':
        provisional = outcome
        outcome = 'insufficient_evidence'
        reasons.append('exploratory')
    if reasons:
        for evidence in assets.values():
            evidence['provisional_outcome'] = evidence['economic_outcome']
            evidence['economic_outcome'] = 'insufficient_evidence'
            evidence['qualification_reasons'] = list(reasons)
    return ResearchReport(protocol_id=protocol.protocol_id, technical_status=technical,
        qualification=protocol.qualification, economic_outcome=outcome, provisional_outcome=provisional,
        reasons=tuple(reasons), assets=assets, units=units)
