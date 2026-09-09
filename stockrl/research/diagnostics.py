"""Pure summaries of already executed decisions; never invoke a policy or RNG."""
from __future__ import annotations

import math

import numpy as np

from stockrl.research.contracts import DiagnosticsReport

LOG_FIELDS = ('entropy_loss', 'approx_kl', 'clip_fraction', 'value_loss', 'explained_variance',
              'learning_rate', 'actual_steps', 'gradient_updates', 'elapsed_seconds')


def _records(value):
    return value.to_dict('records') if hasattr(value, 'to_dict') else list(value)


def _finite(value):
    return value is not None and isinstance(value, (int, float, np.number)) and math.isfinite(value)


def summarize_diagnostics(decisions, *, training_features=None, evaluation_features=None,
                          standardized_features=None, training_log=None, regimes=None):
    """Feature inputs are eight market columns; standardized values precede clipping.

    Regime rows align t momentum/volatility with that decision's t→t+1 return;
    training_volatility_median must be fixed from the training segment.
    """
    rows = _records(decisions)
    unavailable, warnings = [], []
    summary = {'decision_count': len(rows)}

    def ratio(field, predicate):
        values = [r.get(field) for r in rows]
        known = [v for v in values if _finite(v)]
        summary[f'{field}_available_count'] = len(known)
        if len(known) != len(rows) or not rows:
            unavailable.append(field)
            return None
        return sum(predicate(v) for v in known)/len(known)

    boundary = ratio('clipped_action', lambda a: a <= -.999 or a >= .999)
    clipped = ratio('raw_action', lambda a: a < -1 or a > 1)
    near_half = ratio('requested_weight', lambda w: .45 <= w <= .55)
    if boundary is not None and boundary > .9:
        warnings.append('boundary_action_dominance')
    if near_half is not None and near_half > .9:
        warnings.append('near_half_dominance')
    if clipped is not None and clipped > 0:
        warnings.append('output_clipping')
    summary['zero_fill_count'] = sum(r.get('filled_qty') == 0 for r in rows)
    summary['target_actual_difference_count'] = sum(
        _finite(r.get('requested_weight')) and _finite(r.get('executed_weight'))
        and abs(r['requested_weight']-r['executed_weight']) > 1e-12 for r in rows)
    summary['unchanged_position_count'] = sum(
        _finite(a.get('position_weight_close')) and _finite(b.get('position_weight_close'))
        and a['position_weight_close'] == b['position_weight_close'] for a, b in zip(rows, rows[1:]))
    stats = {}
    if all(x is not None for x in (training_features, evaluation_features, standardized_features)):
        train, evaluation, normalized = [np.asarray(x, dtype=float) for x in (training_features, evaluation_features, standardized_features)]
        if any(a.ndim != 2 or a.shape[1] != 8 or not len(a) or not np.isfinite(a).all() for a in (train, evaluation, normalized)) or evaluation.shape != normalized.shape:
            raise ValueError('feature diagnostics require finite, aligned matrices of eight market features')
        names = list(training_features.columns) if hasattr(training_features, 'columns') else list(map(str, range(8)))
        for i, name in enumerate(names):
            low, high = np.quantile(train[:, i], [.01, .99])
            clip_ratio = float(np.mean(np.abs(normalized[:, i]) > 10))
            stats[str(name)] = {
                'clipping_ratio': clip_ratio,
                'outside_training_quantiles_ratio': float(np.mean((evaluation[:, i] < low) | (evaluation[:, i] > high))),
                'training_p01': float(low), 'training_p99': float(high),
                'raw_mean': float(evaluation[:, i].mean()), 'raw_std': float(evaluation[:, i].std()),
                'standardized_mean': float(normalized[:, i].mean()), 'standardized_std': float(normalized[:, i].std()),
                'clipped_mean': float(np.clip(normalized[:, i], -10, 10).mean()),
                'clipped_std': float(np.clip(normalized[:, i], -10, 10).std()),
            }
            if clip_ratio > .05 and 'distribution_shift' not in warnings:
                warnings.append('distribution_shift')
    else:
        unavailable.append('feature_drift')
    groups = {}
    if regimes is None:
        unavailable.append('regimes')
    else:
        grouped = {name: [] for name in ('positive_high', 'positive_low', 'nonpositive_high', 'nonpositive_low')}
        omitted = 0
        for r in _records(regimes):
            required = ('momentum_20', 'volatility_20', 'training_volatility_median', 'log_return_difference', 'position_weight_close', 'fees')
            if not all(_finite(r.get(k)) for k in required):
                omitted += 1
                continue
            name = ('positive' if r['momentum_20'] > 0 else 'nonpositive') + ('_high' if r['volatility_20'] > r['training_volatility_median'] else '_low')
            grouped[name].append(r)
        summary['regime_unavailable_count'] = omitted
        for name, members in grouped.items():
            groups[name] = {'sample_count': len(members), 'low_coverage': len(members) < 20,
                'average_position': float(np.mean([r['position_weight_close'] for r in members])) if members else None,
                'log_return_difference_sum': sum(r['log_return_difference'] for r in members),
                'cost': sum(r.get('cost', r['fees']) for r in members)}
    logs = []
    if training_log is None:
        unavailable.append('training_log')
    else:
        for row in _records(training_log):
            logs.append({k: row.get(k) if _finite(row.get(k)) else None for k in LOG_FIELDS})
    return DiagnosticsReport(boundary_action_ratio=boundary, clipping_ratio=clipped, near_half_ratio=near_half,
        warnings=tuple(warnings), unavailable=tuple(unavailable), feature_stats=stats, regimes=groups,
        training_log=tuple(logs), summary=summary)
