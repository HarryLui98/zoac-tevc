import time
import os
import numpy as np
import gymnasium as gym
import ray
import copy

from algo.utils import *
from algo.models import *
from algo.recording import *
from multi_lane_env.env.idc import *
from multi_lane_env.decision_control.pyth_decision_control import MultiLaneDecisionControl

# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

@ray.remote
class Sampler(object):
    def __init__(self, seed, policy, noise_table, mean, cov_sqrt, label, cfg):
        # initialize environment for each worker
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if cfg['env'] == 'IDCMultiLaneFlowEnv':
            self.env = IDCMultiLaneFlowEnv(seed=seed, label=label)
        else:
            self.env = gym.make(cfg['env'])
        self.current_ego_state = None
        self.current_surr_state = None
        self.current_obs = None
        self.is_done = False
        self.steps = 0
        self.deltas = SharedNoiseTable(noise_table, seed)
        self.policy = copy.deepcopy(policy)
        self.mean = mean
        self.cov_sqrt = cov_sqrt
        self.shift = cfg['reward_shift']
        self.gamma = cfg['gamma']
        self.N_step = cfg['N_step_return']
        self.pim_method = cfg['pim_method']

    def update_param(self, shared_param):
        new_para = copy.deepcopy(shared_param)
        self.policy.set_flat_param(new_para, after_map=True)
        self.mean = new_para

    def init_new_episode(self):
        self.is_done = False
        self.steps = 0
        if isinstance(self.env, IDCMultiLaneFlowEnv):
            self.current_ego_state, self.current_surr_state, self.current_obs = self.env.reset()
        else:
            self.current_obs, _ = self.env.reset()
        if isinstance(self.policy, MultiLaneDecisionControl):
            self.policy.decision_module.reset()
            self.policy.control_module.reset()

    def rollout_several_step(self, n_step_return, noise_idx=0, positive=True, perturbed_param=None):
        # t0 = time.time()
        if self.pim_method == 'zopg':
            direction = self.deltas.get(noise_idx, self.policy.para_num)
            direction = direction if positive else -direction
            # perturbed_param = self.mean + np.dot(self.cov_sqrt, direction)
            perturbed_param = self.mean + self.cov_sqrt * direction
        else:
            perturbed_param = copy.deepcopy(perturbed_param)
        # rollout n steps using perturbed policy
        self.policy.set_flat_param(perturbed_param.astype(np.float32), after_map=True)
        # t1 = time.time()
        # print('set_flat_param time:', t1 - t0)
        if isinstance(self.policy, MultiLaneDecisionControl):
            self.policy.control_module.reset(first=False)
        step = 0
        states = []
        next_states = []
        rewards = []
        not_deads = []
        while step < n_step_return and (not self.is_done):
            # t2 = time.time()
            if isinstance(self.policy, MultiLaneDecisionControl):
                action, policy_info = self.policy.get_action(self.current_ego_state, self.current_surr_state)
            else:
                action = self.policy.get_action(torch.Tensor(self.current_obs))
            # t3 = time.time()
            # print('get_action time:', t3 - t2)
            # t4 = time.time()
            if isinstance(self.env, IDCMultiLaneFlowEnv):
                next_obs, reward, is_done, env_info = self.env.step(action)
                next_ego_state, next_surr_state = env_info['ego_state'], env_info['surr_state']
                not_deads.append(not env_info['dead'])
            else:
                next_obs, reward, terminated, truncated, env_info = self.env.step(action)
                not_deads.append(not terminated)
                is_done = terminated or truncated
            # t5 = time.time()
            # print('step time:', t5 - t4)
            reward -= self.shift
            self.steps += 1
            states.append(torch.Tensor(self.current_obs))
            next_states.append(torch.Tensor(next_obs))
            rewards.append(reward)
            if isinstance(self.env, IDCMultiLaneFlowEnv):
                self.current_ego_state, self.current_surr_state, self.current_obs = next_ego_state, next_surr_state, next_obs
            else:
                self.current_obs = next_obs
            self.is_done = is_done
            step += 1
        addl_info = {}
        return {'states': torch.stack(states), 'next_states': torch.stack(next_states),
                'rewards': torch.Tensor(rewards), 'not_deads': torch.Tensor(not_deads),
                'steps': step, 'is_done': self.is_done, 'noise_idx': noise_idx, 'addl_info': addl_info}
    
    def close_env(self):
        self.env.close()