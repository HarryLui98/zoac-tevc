import os
import time
import ray
import copy
import hydra
from omegaconf import DictConfig, OmegaConf
import json

import numpy as np
import torch
import torch.nn.functional as F
import nevergrad as ng

from algo.utils import *
from algo.sampler import *
from algo.evaluator import *
from algo.recording import *
from algo.optimizers import *
from algo.storage import *
from tensorboardX import SummaryWriter
from multi_lane_env.env.idc import *
from multi_lane_env.decision_control.pyth_decision_control import MultiLaneDecisionControl
import gymnasium as gym

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"

class ZOACLearner(object):
    def __init__(self, cfg):
        self.cfg = cfg
        # self.logdir = cfg['logdir'] + '/' + str(cfg['env']) + '/' + str(cfg['seed'])
        self.logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        self.seed = cfg['seed']
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        configure_output_dir(self.logdir)
        json.dump(OmegaConf.to_container(cfg), open(self.logdir + '/config.json', 'w'), indent=4)
        self.writer = SummaryWriter(self.logdir)
        self.total_steps = 0
        self.total_episodes = 0
        self._init_env()
        self._init_actor_critic()
        self._init_algo_components()

    def _init_env(self):
        if self.cfg['env'] == 'IDCMultiLaneFlowEnv':
            self.env = IDCMultiLaneFlowEnv(seed=self.seed, label=str(self.cfg['label'])+'default')
        else:
            self.env = gym.make(self.cfg['env'])
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.max_action = self.env.action_space.high
        self.min_action = self.env.action_space.low
        self.env.close()
        print("State dim: %i" % self.state_dim)
        print("Action dim: %i" % self.action_dim)

    def _init_actor_critic(self):
        if self.cfg['actor_type'] == "NeuralActor":
            self.policy = NeuralActor(self.state_dim, self.action_dim, 64, self.max_action, self.min_action, self.cfg['seed'])
        elif self.cfg['actor_type'] == "MultiLaneDecisionControl":
            self.policy = MultiLaneDecisionControl(self.seed, self.env.config)
        else:
            raise NotImplementedError
        print("Parameter dim: %i" % self.policy.para_num)
        self.mean = self.policy.get_flat_param(after_map=True)
        # self.mean = np.zeros(self.policy.para_num).astype(np.float32)
        if self.cfg['optimizer'] == 'sgd':
            self.policy_mean_optimizer = SGD(self.mean, lr=self.cfg['actor_mean_step_size'])
        elif self.cfg['optimizer'] == 'adam':
            self.policy_mean_optimizer = Adam(self.mean, lr=self.cfg['actor_mean_step_size'])
        else:
            raise NotImplementedError
        self.critic = Critic(self.state_dim, 1, self.cfg).to(device)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=self.cfg['critic_step_size'])

    def _init_algo_components(self):
        self.deltas_id = create_shared_gaussian_noise.remote(self.seed)
        self.deltas = SharedNoiseTable(ray.get(self.deltas_id), seed=self.seed)
        self.n = self.cfg['samplers_num']
        self.N = self.cfg['N_step_return']
        self.H = self.cfg['train_freq']
        self.B = self.cfg['critic_batch_size']
        self.num_elite = int(self.cfg['elitsm'] * self.n * self.H)
        self.clip = self.cfg['clip_epsilon']
        self.replay_buffer = [ReplayBuffer(self.state_dim, self.N, self.H,
                                                 self.cfg['gamma'], self.cfg['gae_lambda'])
                                    for _ in range(self.n)]
        mean_id = ray.put(self.mean)
        self.sampler_set = [Sampler.remote(seed=self.seed + 3 * i, policy=self.policy, noise_table=self.deltas_id,
                                           mean=mean_id, cov_sqrt = self.cfg['init_paranoise_std'],
                                           label=str(self.cfg['label'])+'sample'+str(self.seed)+str(i), cfg=self.cfg)
                            for i in range(self.n)]
        self.n_eval = self.cfg['evaluators_num']
        self.evaluator_set = [Evaluator.remote(seed=self.seed + 5 * i, policy=self.policy,
                                   num_rollouts=self.cfg['eval_traj_num']//self.n_eval,
                                   label=str(self.cfg['label'])+'eval'+str(self.seed)+str(i), cfg=self.cfg)
                                   for i in range(self.n_eval)]
        if self.cfg['pim_method'] != 'zopg':
            self.tunable_para_high = np.ones(27)
            self.tunable_para_low = - np.ones(27)
            self.tunable_para_init = np.zeros(27)
            self.opt_para = ng.p.Array(init=self.tunable_para_init,
                                       lower=self.tunable_para_low,
                                       upper=self.tunable_para_high)
            opt = getattr(ng.optimizers, self.cfg['pim_method'])
            self.optimizer = opt(parametrization=self.opt_para, budget=self.cfg['max_iter']*self.n*self.H, 
                                 num_workers=self.n*self.H)

    def compute_gae(self):
        for i in range(self.n):
            states, next_states, rewards, dones, alives = self.replay_buffer[i].get_transition()
            if self.cfg['pev_method'] == 'gae':
                with torch.no_grad():
                    if self.cfg['disable_critic']:
                        values = torch.zeros(rewards.shape[0], dtype=torch.float32).to(device)
                        next_values = torch.zeros(rewards.shape[0], dtype=torch.float32).to(device)
                    else:
                        values = self.critic(states.to(device)).squeeze()
                        next_values = self.critic(next_states.to(device)).squeeze()
                    deltas = rewards.to(device) + self.cfg['gamma'] * next_values * alives.to(device) - values
                values, next_values, deltas = values.detach().cpu(), next_values.detach().cpu(), deltas.detach().cpu()
                gaes = torch.zeros(rewards.shape[0], dtype=torch.float32)
                advs = torch.zeros(self.H, dtype=torch.float32)
                discount = (1.0 - dones) * (self.cfg['gamma'] * self.cfg['gae_lambda'])
                traj_k = rewards.shape[0] - 1
                noise_first_idx = torch.nonzero(self.replay_buffer[i].noise_first, as_tuple=True)[0].tolist()
                noise_first_idx.append(states.shape[0])
                assert len(noise_first_idx) == self.H + 1
                traj_gae = 0.0
                for j in range(self.H-1, -1, -1):
                    idx, next_idx = noise_first_idx[j], noise_first_idx[j+1]
                    gae = 0.0
                    for k in range(next_idx-1, idx-1, -1):
                        gae = deltas[k] + discount[k] * gae
                        traj_gae = deltas[k] + discount[k] * traj_gae
                        gaes[traj_k] = traj_gae
                        traj_k -= 1
                    advs[j]= gae
                returns = gaes + values
                if self.cfg['pim_method'] == 'zopg':
                    self.replay_buffer[i].store(values, next_values, returns, advs)
                else:
                    pim_rets = returns[noise_first_idx[:-1]]
                    self.replay_buffer[i].store(values, next_values, returns, pim_rets)
            else:
                assert self.cfg['train_freq'] == 1
                advs = torch.sum(rewards)
                self.replay_buffer[i].advantages[0] = advs
    
    def gen_experience(self, individual_idx):
        t0 = time.time()
        results_id = []
        if self.cfg['pim_method'] == 'zopg':
            if not self.cfg['antithetic']:
                for i in range(self.n):
                    noise_idx = self.deltas.sample_index(self.policy.para_num)
                    results_id.append(self.sampler_set[i].rollout_several_step.remote(self.N, noise_idx, True))
            else:
                assert self.n % 2 == 0
                for i in range(self.n//2):
                    noise_idx = self.deltas.sample_index(self.policy.para_num)
                    results_id.append(self.sampler_set[2*i].rollout_several_step.remote(self.N, noise_idx, True))
                    results_id.append(self.sampler_set[2*i+1].rollout_several_step.remote(self.N, noise_idx, False))
        else:
            for i in range(self.n):
                para = self.optimizer.ask()
                self.current_iter_paras.append(para)
                para_id = ray.put(para.args[0])
                results_id.append(self.sampler_set[i].rollout_several_step.remote(self.N, perturbed_param=para_id))
        # t1 = time.time()
        # print('rollout time:', t1 - t0)
        results = ray.get(results_id)
        # t2 = time.time()
        # print('get results time:', t2 - t1)
        init_id = []
        for i in range(self.n):
            # t3 = time.time()
            result = results[i]
            self.total_steps += result['steps']
            self.replay_buffer[i].add(result)
            # t4 = time.time()
            # print('add to buffer time:', t4 - t3)
            # initialize a new episode when a trajectory is finished, in the corresponding worker
            if self.cfg['episode_type'] == 'Truncated':
                if result['is_done'] is True or individual_idx == self.H - 1:
                    init_id.append(self.sampler_set[i].init_new_episode.remote())
                    self.total_episodes += 1
            elif self.cfg['episode_type'] == 'Full':
                if result['is_done'] is True:
                    init_id.append(self.sampler_set[i].init_new_episode.remote())
                    self.total_episodes += 1
            # t5 = time.time()
            # print('init episode time:', t5 - t4)
        ray.wait(init_id)

    def update_critic(self, iter_num, k):
        if k == 0:
            self.total_obses = []
            for i in range(self.n):
                states = self.replay_buffer[i].get_item('states')
                self.total_obses.append(states)
            self.total_obses = torch.cat(self.total_obses).to(device)
            self.nbatch = self.total_obses.shape[0]
            self.inds = np.arange(self.nbatch)
        self.total_targets = []
        for i in range(self.n):
            targets = self.replay_buffer[i].get_item('returns')
            self.total_targets.append(targets)
        self.total_targets = torch.cat(self.total_targets).to(device)
        # Randomize the indexes
        np.random.shuffle(self.inds)
        # 0 to batch_size with batch_train_size step
        # assert self.nbatch >= self.B
        for start in range(0, self.nbatch - self.B, self.B):
            end = start + self.B
            if self.nbatch - end < self.B:
                end = self.nbatch
            mbinds = self.inds[start:end]
            mb_obs = self.total_obses[mbinds]
            mb_targets = self.total_targets[mbinds]
            self.critic.train()
            mb_estvalue = self.critic(mb_obs).squeeze()
            critic_loss = F.mse_loss(mb_estvalue, mb_targets, reduction='mean')
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

    def update_actor(self, iter_num, m):
        if m == 0:
            self.curr_mean = copy.deepcopy(self.mean)
            self.total_noises = []
            if not self.cfg['antithetic']:
                for i in range(self.n):
                    idxes = self.replay_buffer[i].noise_idxes
                    noises = [self.deltas.get(idx, self.policy.para_num) for idx in idxes]
                    self.total_noises += noises
            else:
                assert self.n % 2 == 0
                for i in range(self.n//2):
                    idxes = self.replay_buffer[2*i].noise_idxes
                    pos_noises = [self.deltas.get(idx, self.policy.para_num) for idx in idxes]
                    self.total_noises += pos_noises
                    neg_noises = [-self.deltas.get(idx, self.policy.para_num) for idx in idxes]
                    self.total_noises += neg_noises
            self.total_noises = np.array(self.total_noises)
        self.total_fits = []
        for i in range(self.n):
            advs = self.replay_buffer[i].advantages
            self.total_fits.append(advs)
        self.total_fits = np.concatenate(self.total_fits, axis=0)
        if self.cfg['pim_method'] == 'zopg':
            self.total_meanw = self.total_noises / self.cfg['init_paranoise_std']
            fits_max_idx = np.argsort(self.total_fits)
            top_fits = self.total_fits[fits_max_idx][-self.num_elite:]
            if self.cfg['fit_norm']:            
                top_fits_mean = np.mean(top_fits) * np.ones(self.num_elite).astype(np.float32)
                top_fits_norm = (top_fits - top_fits_mean) / (np.std(top_fits) + 1e-6)
            else:
                top_fits_norm = top_fits
            top_meanw = self.total_meanw[:self.total_fits.shape[0]][fits_max_idx][-self.num_elite:]
            mean_grad_hat = np.dot(top_fits_norm, top_meanw) / (self.num_elite)
            self.curr_mean, ratio = self.policy_mean_optimizer.update(-mean_grad_hat)
        else:
            for i in range(len(self.total_fits)):
                self.optimizer.tell(self.current_iter_paras[i], -self.total_fits[i])

    def update_workers(self):
        para_id = ray.put(self.mean)
        ray.wait([worker.update_param.remote(para_id) for worker in self.sampler_set])

    def evaluate_policy(self, mean_value):
        mean_id = ray.put(mean_value)
        ray.wait([self.evaluator_set[i].update_param.remote(mean_id, after_map=True) 
                  for i in range(self.n_eval)])
        return [self.evaluator_set[i].eval_rollouts.remote() for i in range(self.n_eval)]
    
    def log_result(self, eval_results, eval_sets):
        rets, steps = [], []
        infos = {
            'cost/speed': [],
            'cost/smooth': [],
            'cost/rule': [],
            'cost/fuel': []
        }
        for i in range(self.n_eval):
            ret, step, info = eval_results[i]
            rets.append(ret)
            steps.append(step)
            for k, _ in infos.items():
                infos[k].append(info[k])
        rets = np.concatenate(rets, axis=0)
        steps = np.concatenate(steps, axis=0)
        log_tabular("Iter", eval_sets['iter_num'])
        log_tabular("Time/Total", eval_sets['time/total'])
        log_tabular("Time/Sample", eval_sets['time/sample'])
        log_tabular("TotalSteps", eval_sets['step'])
        log_tabular("TotalEpisodes", eval_sets['episode'])
        log_tabular("AverageEpsLength", np.mean(steps))
        log_tabular("AverageReturn", np.mean(rets))
        if self.cfg['env'] == 'IDCMultiLaneFlowEnv':
            log_tabular("AverageReturn/Speed", np.mean(np.array(infos['cost/speed'])))
            log_tabular("AverageReturn/Smooth", np.mean(np.array(infos['cost/smooth'])))
            log_tabular("AverageReturn/Rule", np.mean(np.array(infos['cost/rule'])))
            log_tabular("AverageReturn/Fuel", np.mean(np.array(infos['cost/fuel'])))
        log_tabular("StdRewards", np.std(rets))
        log_tabular("MaxRewardRollout", np.max(rets))
        log_tabular("MinRewardRollout", np.min(rets))
        dump_tabular()
        self.writer.add_scalar('Evaluate/AverageReward', np.mean(rets), eval_sets['step'])
        self.writer.add_scalar('Evaluate/AverageEpsLength', np.mean(steps), eval_sets['step'])
        self.writer.add_scalar("Evaluate/StdRewards", np.std(rets), eval_sets['step'])
        policy_save_path = os.path.join(self.logdir, 'policy/')
        critic_save_path = os.path.join(self.logdir, 'critic/')
        np.save(policy_save_path + 'mean_iter_{}'.format(eval_sets['iter_num']+1), eval_sets['mean'])
        if self.cfg['pev_method'] == 'gae':
            value_func_path = os.path.join(critic_save_path, 'value_func_iter_{}.pth'.format(eval_sets['iter_num'] + 1))
            torch.save(eval_sets['critic'], value_func_path)
        if self.cfg['save_addlinfo']:
            np.save(self.logdir + '/addlinfo.npy', self.addlinfo)

    def train(self):
        # start timer
        self.start = time.time()
        self.time_sample = 0
        self.time_actor = 0
        self.time_critic = 0
        ray.wait([sampler.init_new_episode.remote() for sampler in self.sampler_set])
        for iter_num in range(self.cfg['max_iter']):
            # print("Iter = %i" % iter_num)
            self.time_total = time.time() - self.start
            if iter_num % self.cfg['eval_freq'] == 0:
                if iter_num == 0:
                    policy_save_path = os.path.join(self.logdir, 'policy/')
                    if not (os.path.exists(policy_save_path)):
                        os.makedirs(policy_save_path)
                    critic_save_path = os.path.join(self.logdir, 'critic/')
                    if not (os.path.exists(critic_save_path)):
                        os.makedirs(critic_save_path)
                if iter_num != 0:
                    eval_results = ray.get(eval_result_id)
                    self.log_result(eval_results, eval_sets)
                if self.cfg['pim_method'] == 'zopg':
                    recommendation = self.mean
                    eval_result_id = self.evaluate_policy(recommendation)
                else:
                    recommendation = self.optimizer.recommend()
                    eval_result_id = self.evaluate_policy(recommendation.value)
                eval_sets = {'mean': recommendation,
                             'critic': self.critic.state_dict(),
                             'iter_num': iter_num, 'step': self.total_steps, 'episode': self.total_episodes,
                             'time/sample': self.time_sample, 'time/total': self.time_total}
            for i in range(self.cfg['samplers_num']):
                self.replay_buffer[i].reset()
            if self.cfg['pim_method'] == 'zopg':
                self.update_workers()
            else:
                self.current_iter_paras = []
            # Sample rollouts
            time1 = time.time()
            for j in range(self.cfg['train_freq']):
                self.gen_experience(j)
            time2 = time.time()
            self.time_sample += time2 - time1
            self.compute_gae()
            # First-order PEV
            if self.cfg['pev_method'] == 'gae' and not self.cfg['disable_critic']:
                for k in range(self.cfg['critic_epochs']):
                    if self.cfg['recompute_advs']:
                        self.compute_gae()
                    self.update_critic(iter_num, k)
            # Zeroth-order PIM
            for m in range(self.cfg['actor_epochs']):
                self.update_actor(iter_num, m)
            self.mean = self.curr_mean
        self.writer.close()
        ray.wait([self.sampler_set[i].close_env.remote() for i in range(self.n)])
        ray.wait([self.evaluator_set[i].close_env.remote() for i in range(self.n_eval)])
