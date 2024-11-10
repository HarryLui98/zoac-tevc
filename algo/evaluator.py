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
import matplotlib.pyplot as plt
from multi_lane_env.decision_control.pyth_decision_control import MultiLaneDecisionControl

# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

@ray.remote
class Evaluator(object):
    def __init__(self, seed, policy, num_rollouts, label, cfg):
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if cfg['env'] == 'IDCMultiLaneFlowEnv':
            self.env = IDCMultiLaneFlowEnv(seed=seed, label=label)
        else:
            self.env = gym.make(cfg['env'])
        self.policy = copy.deepcopy(policy)
        self.num_rollouts = num_rollouts

    def update_param(self, shared_param, after_map=True):
        new_para = shared_param.copy()
        self.policy.set_flat_param(new_para, after_map)

    def eval_rollout_once(self):
        total_reward = 0
        total_step = 0
        if isinstance(self.env, IDCMultiLaneFlowEnv):
            ego_state, surr_state, obs = self.env.reset()
        else:
            obs, _ = self.env.reset()
        is_done = False
        if isinstance(self.policy, MultiLaneDecisionControl):
            self.policy.decision_module.reset()
            self.policy.control_module.reset()
        info = {
            'cost/speed': 0,
            'cost/smooth': 0,
            'cost/rule': 0,
            'cost/fuel': 0
        }
        while not is_done:
            if isinstance(self.policy, MultiLaneDecisionControl):
                action, policy_info = self.policy.get_action(ego_state, surr_state)
            else:
                action = self.policy.get_action(torch.Tensor(obs))
            if isinstance(self.env, IDCMultiLaneFlowEnv):
                next_obs, reward, is_done, env_info = self.env.step(action)
                next_ego_state, next_surr_state = env_info['ego_state'], env_info['surr_state']
                ego_state, surr_state, obs = next_ego_state, next_surr_state, next_obs
                for k, v in env_info.items():
                    if 'cost' in k:
                        info[k] += v
            else:
                next_obs, reward, terminated, truncated, env_info = self.env.step(action)
                is_done = terminated or truncated
                obs = next_obs
            total_reward += reward
            total_step += 1
        return total_step, total_reward, info

    def eval_rollouts(self):
        # self.policy.eval()
        rets = []
        steps = []
        infos = {
                'cost/speed': [],
                'cost/smooth': [],
                'cost/rule': [],
                'cost/fuel': []
            }
        for _ in range(self.num_rollouts):
            step, ret, info = self.eval_rollout_once()
            rets.append(ret)
            steps.append(step)
            for k, v in info.items():
                infos[k].append(v)
        for k, v in infos.items():
            infos[k] = np.array(v)
        return np.array(rets), np.array(steps), infos
    
    def eval_rollouts_store(self, n_rollouts, store_base_path):
        # self.policy.eval()
        decision_stat = np.zeros((5, n_rollouts))
        for episode in range(n_rollouts):
            if not (os.path.exists(store_base_path+'/allinfo/' + str(episode) + '/')):
                os.makedirs(store_base_path+'/allinfo/' + str(episode) + '/')
            if not (os.path.exists(store_base_path+'/selectvar/' + str(episode) + '/')):
                os.makedirs(store_base_path+'/selectvar/' + str(episode) + '/')
            # if not (os.path.exists(store_base_path+'/fig/' + str(episode) + '/')):
            #     os.makedirs(store_base_path+'/fig/' + str(episode) + '/')
            ego_state, surr_state, obs = self.env.reset()
            is_done = False
            self.policy.decision_module.reset()
            self.policy.control_module.reset()
            step, ret = 0, 0
            u_sum, axsquare_sum, aysquare_sum = 0, 0, 0
            fuel = 0
            while not is_done:
                action, policy_info = self.policy.get_action(ego_state, surr_state)
                next_obs, reward, is_done, env_info = self.env.step(action)
                fuel -= env_info['cost/fuel'] / 1000 * 0.1 # L
                next_ego_state, next_surr_state = env_info['ego_state'], env_info['surr_state']
                target_lane_num = policy_info['target_lane_num']
                cf_state = policy_info['cf_state']
                target_y = (-4 + target_lane_num + 0.5) * 3.75
                x, u, v, yaw, y, phi = ego_state
                next_x, next_u, next_v, next_yaw, next_y, next_phi = next_ego_state
                steer, acc, _ = action
                delta_y = y - target_y
                delta_phi_deg = phi * 180 / np.pi
                front_d = cf_state[0] - x
                steer_deg = steer * 180 / np.pi
                axsq = (next_u-u) ** 2 / 0.01
                aysq = (next_v - v) ** 2 / 0.01
                u_sum += u
                axsquare_sum += axsq
                aysquare_sum += aysq
                selectvar = np.array([u, yaw, delta_y, delta_phi_deg, front_d, steer_deg, acc, axsq, aysq])
                ego_state, surr_state, obs = next_ego_state, next_surr_state, next_obs
                step += 1
                ret += reward
                # if step % 10 == 0:
                #     self.env.render()
                #     plt.savefig(store_base_path + '/fig/' + str(episode) + '/' + str(step) + '.jpg')
                np.savez(store_base_path + '/allinfo/' + str(episode) + '/' + str(step) + '.npz', ego_state=ego_state, cf_state=cf_state,
                    action=action, surr_state=surr_state)
                np.save(store_base_path + '/selectvar/' + str(episode) + '/' + str(step) + '.npy', selectvar)
            decision_stat[0, episode] = ret
            decision_stat[1, episode] = step
            u_mean = 3.6 * u_sum / step # km/h
            i_comfort = 1.4 * np.sqrt(axsquare_sum / step + aysquare_sum / step)
            decision_stat[2, episode] = u_mean
            decision_stat[3, episode] = i_comfort
            decision_stat[4, episode] = 100 * fuel / (u_mean * step * 0.1 / 3600) # L/100km
        if not (os.path.exists(store_base_path)):
                os.makedirs(store_base_path)
        np.save(store_base_path + '/decisionstat.npy', decision_stat)        
    
    def close_env(self):
        self.env.close()
