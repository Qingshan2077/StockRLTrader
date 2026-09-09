"""Independent v2 learning, validation-only selection and closed-loop stress replay."""
from __future__ import annotations
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import math
import random
import shutil
import time
import numpy as np

from stockrl.control import ExecutionControl
from stockrl.research.baselines import REFERENCE_IDS, reference_weight, match_fixed_weight, baseline_cache_key
from stockrl.research.contracts import CheckpointResult, CostResult, UnitResult
from stockrl.research.folds import rollout_steps


def _json(path, value):
    if hasattr(value,'model_dump'): value=value.model_dump(mode='json')
    Path(path).write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False),encoding='utf-8')


def cost_profile(profile, scenario):
    multiplier={'base':1,'execution_x2':2,'execution_x3':3}[scenario]
    def scaled(item):
        values={name:getattr(item,name)*multiplier for name in ('commission_rate','commission_min','slippage')}
        if not all(math.isfinite(v) and v>=0 for v in values.values()) or values['slippage']>=1 or values['commission_rate']>=1:
            raise ValueError('invalid stressed execution costs')
        return replace(item,**values)
    result=scaled(profile)
    return replace(result,fee_intervals=tuple(scaled(item) for item in profile.fee_intervals))


def select_checkpoint(candidates):
    if not candidates: raise ValueError('no trained checkpoints')
    return min(candidates,key=lambda item:(-item['validation_log_return'],item['actual_steps']))


def predict_with_diagnostics(model, observation):
    """Use the actual deterministic policy output, without another random draw."""
    import torch
    tensor,_=model.policy.obs_to_tensor(observation)
    model.policy.set_training_mode(False)
    with torch.no_grad():
        if hasattr(model.policy,'get_distribution'):
            distribution=model.policy.get_distribution(tensor)
            raw=distribution.get_actions(deterministic=True).cpu().numpy().reshape(-1)
            normal=distribution.distribution
            extra={'policy_mean':float(normal.mean.cpu().numpy().reshape(-1)[0]),
                   'policy_std':float(normal.stddev.cpu().numpy().reshape(-1)[0])}
        else:
            raw=model.policy.actor(tensor,deterministic=True).cpu().numpy().reshape(-1)
            extra={'policy_mean':None,'policy_std':None}
    clipped=np.clip(raw,model.action_space.low,model.action_space.high)
    return clipped,dict(raw_action=float(raw[0]),clipped_action=float(clipped[0]),**extra)


def train_candidates(env,budget,seed,locked_steps,output,evaluate,control=None):
    """Evaluate hooks run after train(), never the pre-update rollout callback."""
    import torch
    from stable_baselines3 import PPO,SAC
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.utils import set_random_seed
    control=control or ExecutionControl()
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    locked_steps=tuple(locked_steps)
    from stockrl.research.folds import checkpoint_steps
    if not locked_steps or locked_steps != checkpoint_steps(budget,len(locked_steps)):
        # SAC excludes pre-learning candidates, so its remaining count need not regenerate the schedule.
        valid=any(locked_steps==checkpoint_steps(budget,n) for n in range(1,21))
        if not valid: raise ValueError('checkpoint schedule differs from locked budget')
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    set_random_seed(seed,using_cuda=False)
    candidates=[]; logs=[]; start=time.monotonic()
    def after_update(model):
        control.check()
        data=model.logger.name_to_value
        row={name:(float(data['train/'+name]) if 'train/'+name in data and math.isfinite(float(data['train/'+name])) else None)
             for name in ('entropy_loss','approx_kl','clip_fraction','value_loss','explained_variance','learning_rate')}
        row.update(actual_steps=int(model.num_timesteps),gradient_updates=int(model._n_updates),elapsed_seconds=time.monotonic()-start)
        logs.append(row)
        if model.num_timesteps not in locked_steps: return
        name=f'checkpoint_{model.num_timesteps}.zip'; model.save(output/name)
        states=(random.getstate(),np.random.get_state(),torch.random.get_rng_state())
        try: result=evaluate(model,int(model.num_timesteps))
        finally:
            random.setstate(states[0]); np.random.set_state(states[1]); torch.random.set_rng_state(states[2])
        candidates.append(dict(**result,actual_steps=int(model.num_timesteps),gradient_updates=int(model._n_updates),
             model_path=name,model_sha256=hashlib.sha256((output/name).read_bytes()).hexdigest(),
             evaluated_at=datetime.now(timezone.utc).isoformat()))
    class PostUpdate:
        def train(self,*args,**kwargs):
            result=super().train(*args,**kwargs)
            after_update(self)
            return result
    class ResearchPPO(PostUpdate,PPO): pass
    class ResearchSAC(PostUpdate,SAC): pass
    class CheckControl(BaseCallback):
        def _on_step(self):
            control.emit({'event':'training','seed':seed,'timesteps':self.num_timesteps},progress=True)
            return True
    common=dict(policy='MlpPolicy',env=Monitor(env),seed=seed,device='cpu',verbose=0,policy_kwargs={'net_arch':[32,32]})
    if budget.algorithm=='PPO':
        rollout=rollout_steps(budget)
        batch=max(s for s in range(2,min(32,rollout)+1) if rollout%s==0)
        model=ResearchPPO(**common,n_steps=rollout,batch_size=batch,n_epochs=4)
    else:
        model=ResearchSAC(**common,buffer_size=max(1000,min(1000000,budget.requested_timesteps)),
            learning_starts=min(100,budget.requested_timesteps//4),batch_size=max(2,min(32,budget.requested_timesteps//4)),train_freq=1,gradient_steps=1)
    try: model.learn(total_timesteps=budget.requested_timesteps,callback=CheckControl())
    finally: model.get_env().close()
    if tuple(c['actual_steps'] for c in candidates)!=locked_steps: raise RuntimeError('not all locked checkpoints evaluated')
    return model,candidates,logs


_WINDOW_CACHE=[]


def window_bundle(bundle,window):
    """Expose this reward segment and at most its 252 explicitly locked past sessions."""
    first=(window.warmup_sessions[-252:] or (window.initial_session,))[0]
    last=window.reward_sessions[-1]
    if not any(bar.session == first for bar in bundle.bars):
        # Retain a valuation anchor across a suspension. This adds no reward or
        # learner observation; the locked window/warmup remains unchanged.
        anchors = [bar.session for bar in bundle.bars if bar.session < first]
        if anchors:
            first = anchors[-1]
    if bundle.sessions[0].session==first and bundle.sessions[-1].session==last:
        return bundle
    for original,bounds,prepared in _WINDOW_CACHE:
        if original is bundle and bounds==(first,last): return prepared
    prepared=replace(bundle,bars=tuple(b for b in bundle.bars if first<=b.session<=last),
        sessions=tuple(s for s in bundle.sessions if first<=s.session<=last),
        tradability=tuple(s for s in bundle.tradability if first<=s.session<=last),
        actions=tuple(a for a in bundle.actions if first<=a.effective_session<=last))
    _WINDOW_CACHE.append((bundle,(first,last),prepared))
    del _WINDOW_CACHE[:-4]
    return prepared


def evaluate_strategy(bundle,window,normalizer,*,model=None,strategy_id=None,matched_weight=None,control=None,key=None,checkpoint_id=None):
    from stockrl.env_v2 import TradingEnvV2
    from stockrl.research.report import calculate_metrics
    control=control or ExecutionControl()
    env=TradingEnvV2(window_bundle(bundle,window),window,normalizer=normalizer)
    obs,_=env.reset()
    try:
        while True:
            control.check()
            extra={}
            if model is not None:
                action,extra=predict_with_diagnostics(model,obs)
            else:
                w=reference_weight(strategy_id,env.feature_prices.Close.iloc[:env.current_step+1],
                    env.sessions[env.current_step+1].session,env.current_session,matched_weight=matched_weight)
                action=None if w is None else np.asarray([2*w-1],dtype=np.float32)
            obs,_,terminated,truncated,_=env.step(action)
            if model is not None:
                env.decision_logs[-1].update(extra,raw_action_reason=None)
            if key is not None:
                env.decision_logs[-1].update(seed=key.seed,fold_id=key.fold_id,checkpoint_id=checkpoint_id)
            if terminated or truncated: break
        return {'history':env.history,'trades':env.trades,'decisions':env.decision_logs,
                'metrics':calculate_metrics(env.history,env.trades).model_dump(mode='json')}
    finally: env.close()


def _cached_reference(cache,key,compute):
    path=cache/(key+'.json')
    if path.exists():
        try:
            envelope=json.loads(path.read_text(encoding='utf-8'))
            payload=json.dumps(envelope['result'],sort_keys=True,separators=(',',':'),allow_nan=False)
            if envelope['key']==key and envelope['sha256']==hashlib.sha256(payload.encode()).hexdigest():
                return envelope['result']
        except (ValueError,KeyError,TypeError): pass
    result=compute()
    payload=json.dumps(result,sort_keys=True,separators=(',',':'),allow_nan=False)
    cache.mkdir(parents=True,exist_ok=True)
    _json(path,{'key':key,'sha256':hashlib.sha256(payload.encode()).hexdigest(),'result':result})
    return result


def run_unit(protocol,key,output,control,*,bundle=None):
    from stable_baselines3 import PPO,SAC
    from stockrl.env_v2 import TradingEnvV2
    from stockrl.market.features import NormalizerV2,build_features,build_feature_prices
    from stockrl.market.contracts import to_dict
    from stockrl.research.diagnostics import summarize_diagnostics
    from stockrl.research.contracts import StrategyMetrics
    import pandas as pd
    if bundle is None: raise ValueError('run_unit requires a verified market bundle')
    if key.instrument_id not in protocol.instrument_ids or key.seed not in protocol.seeds:
        raise ValueError('unit is outside the locked protocol')
    folds=[f for f in protocol.fold_plan if f.fold_id==key.fold_id and f.instrument_id in (None,key.instrument_id)]
    if len(folds)!=1: raise ValueError('unit fold is missing or ambiguous')
    fold=folds[0]; output=Path(output); output.mkdir(parents=True,exist_ok=True)
    control=control or ExecutionControl(); control.check()
    train_bundle=window_bundle(bundle,fold.train)
    validation_bundle=window_bundle(bundle,fold.validation)
    train_features=build_features(train_bundle).loc[list(fold.train.reward_sessions)]
    normalizer=NormalizerV2.fit(train_features); normalizer.save(output/'normalizer.json')
    checkpoint_dir=output/'checkpoints'
    control.phase('training',**key.model_dump())
    def validate(model,steps):
        control.phase('validating',**key.model_dump(),actual_steps=steps)
        replay=evaluate_strategy(validation_bundle,fold.validation,normalizer,model=model,control=control)
        _json(checkpoint_dir/f'validation_{steps}.json',replay)
        control.phase('training',**key.model_dump())
        return dict(validation_log_return=math.log(replay['history'][-1]['nav']/replay['history'][0]['nav']),
                    validation_reward=sum(r['reward'] for r in replay['history'][1:]),metrics=replay['metrics'])
    model,records,logs=train_candidates(TradingEnvV2(train_bundle,fold.train,normalizer=normalizer,
             episode_length=protocol.training_budget.episode_length),protocol.training_budget,key.seed,
             protocol.checkpoint_steps,checkpoint_dir,validate,control)
    selected=select_checkpoint(records)
    shutil.copyfile(checkpoint_dir/selected['model_path'],output/'model.zip')
    learner={'PPO':PPO,'SAC':SAC}[protocol.training_budget.algorithm]
    selected_model=learner.load(output/'model.zip',device='cpu')
    checkpoint_id=f"checkpoint_{selected['actual_steps']}"
    candidates=tuple(CheckpointResult(checkpoint_id=f"checkpoint_{r['actual_steps']}",
        **{k:r[k] for k in ('actual_steps','gradient_updates','model_sha256','validation_log_return','evaluated_at','metrics','validation_reward')}) for r in records)
    calibration=match_fixed_weight(selected['metrics']['annualized_volatility'],lambda w:
        evaluate_strategy(validation_bundle,fold.validation,normalizer,strategy_id='matched_fixed',matched_weight=w,control=control)['metrics']['annualized_volatility'])
    selection=dict(selected_checkpoint_id=checkpoint_id,selection_reason='maximum_complete_validation_log_return; earlier_steps_on_tie',
                   candidates=[c.model_dump(mode='json') for c in candidates],calibration=calibration,
                   requested_steps=protocol.training_budget.requested_timesteps,actual_steps=int(model.num_timesteps))
    _json(output/'checkpoint_selection.json',selection)
    _json(output/'training_log.json',logs)
    costs=[]; base_replays=None
    # Include actual immutable input content as well as supplied source fingerprints.
    content_fingerprint=hashlib.sha256(json.dumps(to_dict(bundle),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    cache=output.parent/'_baseline_cache'
    for scenario in protocol.cost_scenarios:
        control.phase('testing',**key.model_dump(),scenario=scenario)
        stressed=replace(bundle,market_profile=cost_profile(bundle.market_profile,scenario))
        replays={'rl':evaluate_strategy(stressed,fold.test,normalizer,model=selected_model,control=control,key=key,checkpoint_id=checkpoint_id)}
        for strategy_id in (*REFERENCE_IDS,'matched_fixed'):
            cache_key=baseline_cache_key(dataset_fingerprint=content_fingerprint,market_profile=stressed.market_profile,
                window=fold.test,strategy_id=strategy_id,seed=key.seed,matched_weight=calibration['weight'],protocol_fingerprint=protocol.lock().sha256)
            replays[strategy_id]=_cached_reference(cache,cache_key,lambda sid=strategy_id:
                evaluate_strategy(stressed,fold.test,None,strategy_id=sid,matched_weight=calibration['weight'],control=control))
            for decision in replays[strategy_id]['decisions']:
                decision.update(seed=key.seed,fold_id=key.fold_id,checkpoint_id=None)
        paths={}
        for name,replay in replays.items():
            directory=output/scenario/name; directory.mkdir(parents=True,exist_ok=True)
            for artifact,contents in replay.items():
                target=directory/(('report' if artifact=='metrics' else artifact)+'.json')
                _json(target,contents); paths[f'{name}/{artifact}']=target.relative_to(output).as_posix()
        metrics={name:StrategyMetrics.model_validate(replay['metrics']) for name,replay in replays.items()}
        rv,mv=metrics['rl'].annualized_volatility,metrics['matched_fixed'].annualized_volatility
        comparable=abs(rv-mv)<=max(.02,.2*mv) if rv is not None and mv is not None else False
        costs.append(CostResult(scenario_id=scenario,model_sha256=selected['model_sha256'],metrics=metrics,risk_comparable=comparable,artifact_paths=paths))
        if scenario=='base': base_replays=replays
    features=build_features(bundle)
    decision_dates=[r['decision_session'] for r in base_replays['rl']['decisions']]
    evaluation=features.loc[decision_dates]
    standardized=(evaluation-normalizer.mean)/normalizer.scale
    prices=build_feature_prices(bundle).Close
    volatility=prices.pct_change().rolling(20).std(ddof=1)*np.sqrt(252)
    momentum=prices/prices.shift(20)-1
    train_vol=volatility.loc[list(fold.train.reward_sessions)].dropna()
    threshold=float(train_vol.median()) if len(train_vol) else None
    regimes=[]
    for rl,ref in zip(base_replays['rl']['history'][1:],base_replays['matched_fixed']['history'][1:]):
        i=len(regimes); day=decision_dates[i]
        r0=base_replays['rl']['history'][i]['nav']; b0=base_replays['matched_fixed']['history'][i]['nav']
        regimes.append(dict(momentum_20=float(momentum.loc[day]) if pd.notna(momentum.loc[day]) else None,
            volatility_20=float(volatility.loc[day]) if pd.notna(volatility.loc[day]) else None,
            training_volatility_median=threshold,log_return_difference=math.log(rl['nav']/r0)-math.log(ref['nav']/b0),
            position_weight_close=rl['position_weight_close'],fees=rl['fees'],cost=rl['cost']))
    diagnostics=summarize_diagnostics(base_replays['rl']['decisions'],training_features=train_features,evaluation_features=evaluation,
                 standardized_features=standardized,training_log=logs,regimes=regimes)
    validation_dates = [fold.validation.initial_session, *fold.validation.reward_sessions[:-1]]
    validation_features = features.loc[validation_dates]
    validation_drift = summarize_diagnostics([], training_features=train_features,
        evaluation_features=validation_features,
        standardized_features=(validation_features-normalizer.mean)/normalizer.scale)
    diagnostics = diagnostics.model_copy(update={
        'feature_stats': {**diagnostics.feature_stats, **{'validation/'+name: value for name, value in validation_drift.feature_stats.items()}},
        'warnings': (*diagnostics.warnings, *('validation_'+warning for warning in validation_drift.warnings)),
        'summary': {**diagnostics.summary, 'feature_windows': {'unprefixed': 'test', 'validation/': 'validation'}}})
    _json(output/'diagnostics.json',diagnostics)
    result=UnitResult(key=key,cost_results=tuple(costs),candidates=candidates,selected_checkpoint_id=checkpoint_id,
            selection_reason=selection['selection_reason'],matched_weight=calibration['weight'],calibration_error=calibration['calibration_error'],actual_steps=int(model.num_timesteps))
    _json(output/'unit_result.json',result)
    return result
