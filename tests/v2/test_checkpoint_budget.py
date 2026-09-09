import gymnasium as gym
import numpy as np
import pytest
from stockrl.research.runner import train_candidates, select_checkpoint, predict_with_diagnostics
from stockrl.research.contracts import TrainingBudget
from stockrl.research.folds import checkpoint_steps


class TinyEnv(gym.Env):
    observation_space = gym.spaces.Box(-1,1,(12,),dtype=np.float32)
    action_space = gym.spaces.Box(-1,1,(1,),dtype=np.float32)
    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        self.i=0
        return np.zeros(12,dtype=np.float32), {}
    def step(self, action):
        self.i+=1
        return np.zeros(12,dtype=np.float32), float(action[0]), self.i==8, False, {}


@pytest.mark.parametrize('algorithm', ['PPO','SAC'])
def test_candidates_are_exact_locked_post_update_models(tmp_path,algorithm):
    budget=TrainingBudget(requested_timesteps=16,algorithm=algorithm)
    observed=[]
    def evaluate(model, step):
        observed.append((step,model._n_updates))
        return {'score': float(-step)}
    model, records, logs = train_candidates(TinyEnv(),budget,42,checkpoint_steps(budget,3),tmp_path,evaluate)
    assert tuple(s for s,n in observed)==checkpoint_steps(budget,3)
    assert all(n>0 for s,n in observed)
    assert model.num_timesteps==checkpoint_steps(budget,3)[-1]
    assert len(logs)>0
    assert all((tmp_path/r['model_path']).exists() for r in records)


def test_checkpoint_ties_select_earlier_and_ignore_test_values():
    assert select_checkpoint([{'actual_steps':20,'validation_log_return':.1}, {'actual_steps':10,'validation_log_return':.1}])['actual_steps']==10


def test_deterministic_diagnostics_do_not_advance_rng(tmp_path):
    import torch
    from stable_baselines3 import PPO
    model=PPO('MlpPolicy',TinyEnv(),n_steps=2,batch_size=2,policy_kwargs={'net_arch':[32,32]},seed=42)
    before=torch.random.get_rng_state().clone()
    action, info=predict_with_diagnostics(model,np.zeros(12,dtype=np.float32))
    assert torch.equal(before,torch.random.get_rng_state())
    assert info['raw_action'] is not None and info['policy_std']>0
    assert action[0]==np.clip(info['raw_action'],-1,1)


def test_validation_diagnostics_cannot_perturb_training_rng(tmp_path):
    import torch
    budget=TrainingBudget(requested_timesteps=16)
    def noisy_eval(model,step):
        torch.rand(100)
        np.random.random(100)
        return {'score':0}
    clean,_,_=train_candidates(TinyEnv(),budget,42,checkpoint_steps(budget,2),tmp_path/'clean',lambda m,s:{'score':0})
    noisy,_,_=train_candidates(TinyEnv(),budget,42,checkpoint_steps(budget,2),tmp_path/'noisy',noisy_eval)
    assert all(torch.equal(v,noisy.policy.state_dict()[k]) for k,v in clean.policy.state_dict().items())


def test_cancelled_training_cannot_publish_selected_checkpoint(tmp_path):
    from stockrl.control import ExecutionControl,ExperimentCancelled
    budget=TrainingBudget(requested_timesteps=16)
    with pytest.raises(ExperimentCancelled):
        train_candidates(TinyEnv(),budget,42,checkpoint_steps(budget,2),tmp_path,lambda m,s:{'score':0},ExecutionControl(cancel_requested=lambda:True))
    assert not list(tmp_path.glob('*.zip'))
