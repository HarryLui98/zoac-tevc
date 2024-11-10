import numpy as np
import argparse
import os
import time
import parser
import copy
import torch.nn.functional as F
import nevergrad as ng
import ray

from multi_lane_env.decision_control.pyth_decision_control import MultiLaneDecisionControl
from algo.evaluator import *
from algo.recording import *
from tensorboardX import SummaryWriter
from concurrent import futures
import datetime


class ControllerCostMinimizer(object):
    def __init__(self, seed, env_name, opt_name, budget, worker_num, evaluator_num, logdir):
        random.seed(seed)
        np.random.seed(seed)
        if env_name == 'IDCMultiLaneFlowEnv':
            self.env = IDCMultiLaneFlowEnv(seed, opt_name + str(seed))
            self.tunable_para_high = np.ones(27)
            self.tunable_para_low = - np.ones(27)
            self.tunable_para_init = np.zeros(27)
            self.opt_para = ng.p.Array(init=self.tunable_para_init,
                                       lower=self.tunable_para_low,
                                       upper=self.tunable_para_high)
            self.policy = MultiLaneDecisionControl(seed, self.env.config)
        else:
            self.env = gym.make(env_name)
            self.state_dim = self.env.observation_space.shape[0]
            self.action_dim = self.env.action_space.shape[0]
            self.max_action = self.env.action_space.high
            self.min_action = self.env.action_space.low
            self.env.close()
            self.policy = NeuralActor(self.state_dim, self.action_dim, 64, self.max_action, self.min_action, seed)
            self.tunable_para_init = self.policy.get_flat_param()
            self.opt_para = ng.p.Array(init=self.tunable_para_init)
        
        self.trainer = [Evaluator.remote(seed=seed + 3 * i, policy=self.policy, num_rollouts=1, 
                                         label=opt_name + '_' + str(seed) + '_sample' + str(i),
                                         cfg={'env': env_name})
                                        for i in range(worker_num)]
        self.evaluator = [Evaluator.remote(seed=seed + 5 * i, policy=self.policy, num_rollouts=2, 
                                           label=opt_name + '_' + str(seed) + '_eval' + str(i),
                                           cfg={'env': env_name})
                                        for i in range(evaluator_num)]
        self.worker_num = worker_num
        self.n_eval = evaluator_num
        self.budget = budget
        opt = getattr(ng.optimizers, opt_name)
        self.optimizer = opt(parametrization=self.opt_para, budget=budget, num_workers=worker_num)
        self.logdir = logdir
        configure_output_dir(logdir)
        self.writer = SummaryWriter(logdir)
        self.total_steps = 0
        self.total_episodes = 0

    def fitness(self, para, *args, **kwargs):
        thread_id = kwargs['thread_id']
        para_id = ray.put(para)
        ray.wait([self.trainer[thread_id].update_param.remote(para_id, after_map=True)])
        result_id= self.trainer[thread_id].eval_rollouts.remote()
        return result_id

    def evaluate(self, current_para):
        para_id = ray.put(current_para)
        self.policy.set_flat_param(current_para, after_map=True)
        ray.wait([self.evaluator[i].update_param.remote(para_id, after_map=True) for i in range(self.n_eval)])
        return [self.evaluator[i].eval_rollouts.remote() for i in range(self.n_eval)]
    
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
        log_tabular("AverageReturn/Speed", np.mean(np.array(infos['cost/speed'])))
        log_tabular("AverageReturn/Smooth", np.mean(np.array(infos['cost/smooth'])))
        log_tabular("AverageReturn/Rule", np.mean(np.array(infos['cost/rule'])))
        log_tabular("AverageReturn/Fuel", np.mean(np.array(infos['cost/fuel'])))
        log_tabular("StdRewards", np.std(rets))
        log_tabular("MaxRewardRollout", np.max(rets))
        log_tabular("MinRewardRollout", np.min(rets))
        dump_tabular()
        self.writer.add_scalar('Evaluate/AverageReward', np.mean(rets), self.total_steps)
        self.writer.add_scalar('Evaluate/AverageEpsLength', np.mean(steps), self.total_steps)
        self.writer.add_scalar("Evaluate/StdRewards", np.std(rets), self.total_steps)
        policy_save_path = os.path.join(self.logdir, 'policy/')
        if not (os.path.exists(policy_save_path)):
                        os.makedirs(policy_save_path)
        np.save(policy_save_path + 'mean_iter_{}'.format(eval_sets['iter_num']+1), eval_sets['mean'])

    def optimize(self):
        self.start = time.time()
        self.time_sample = 0
        for u in range(self.budget // self.worker_num):
            if self.total_steps > 12e5:
                break
            if u % 5 == 0:
                if u != 0:
                    eval_results = ray.get(eval_result_id)
                    self.log_result(eval_results, eval_sets)
                recommendation = self.optimizer.recommend()
                eval_result_id = self.evaluate(recommendation.value)
                self.total_time = time.time() - self.start
                eval_sets = {'mean': recommendation.value,
                            'iter_num': u, 'step': self.total_steps, 'episode': self.total_episodes,
                            'time/sample': self.time_sample, 'time/total': self.total_time}
            x = []
            for _ in range(self.worker_num):
                x.append(self.optimizer.ask())
            y, c = [], []
            time1 = time.time()
            for i in range(self.worker_num):
                result_id = self.fitness(*x[i].args, **x[i].kwargs, thread_id=i)
                y.append(result_id)
            results = ray.get(y)
            for i in range(self.worker_num):
                rets, steps, _ = results[i]
                ret = np.mean(rets)
                step = np.sum(steps)
                c.append(-ret)
                self.total_steps += step
                self.total_episodes += 1
            time2 = time.time()
            self.time_sample += time2 - time1
            for thread_id in range(self.worker_num):
                self.optimizer.tell(x[thread_id], c[thread_id])            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt_name', type=str, default='CMA') # 'CMA', 'PSO', 'TwoPointsDE'
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--env_name', type=str, default='IDCMultiLaneFlowEnv')
    args = parser.parse_args()
    algo_params = vars(args)
    ray.init(local_mode=False, include_dashboard=False)
    learner = ControllerCostMinimizer(seed=algo_params['seed'], env_name=algo_params['env_name'], opt_name=algo_params['opt_name'], budget=1000, worker_num=8,
                                      evaluator_num=4, logdir='./baseline/' + algo_params['opt_name'] + '/' + algo_params['env_name'] + '/' + str(algo_params['seed']))
    learner.optimize()
    learner.env.close()
    ray.wait([learner.evaluator[i].close_env.remote() for i in range(4)])
    ray.wait([learner.trainer[i].close_env.remote() for i in range(8)]) 
    ray.shutdown()
