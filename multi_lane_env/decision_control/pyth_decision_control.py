import random
import math
import numpy as np
import casadi
import gym
from multi_lane_env.decision_control.pyth_idm_mobil import BehaviorDecision_IDM_MOBIL
from multi_lane_env.decision_control.pyth_planning import Planning_IDC
from multi_lane_env.decision_control.pyth_tracking_mpc import ModelPredictiveController


class MultiLaneDecisionControl(object):
    def __init__(self, seed, config):
        random.seed(seed)
        np.random.seed(seed)
        # self.env = env
        self.decision_module = BehaviorDecision_IDM_MOBIL(seed, config)
        self.planning_module = Planning_IDC(seed, config)
        self.control_module = ModelPredictiveController(seed, config)
        self.decision_para_num = self.decision_module.para_num
        self.control_para_num = self.control_module.para_num
        self.para_num = self.decision_para_num + self.control_para_num

    def get_flat_param(self, after_map=True):
        decision_para = self.decision_module.get_flat_param(after_map)
        control_para = self.control_module.get_flat_param(after_map)
        return np.concatenate([decision_para, control_para])

    def set_flat_param(self, para, after_map=True):
        self.decision_module.set_flat_param(para[:self.decision_para_num], after_map)
        self.planning_module.set_flat_param(para[:self.decision_para_num], after_map)
        self.control_module.set_flat_param(para[self.decision_para_num:], after_map)

    def set_expert_param(self):
        self.decision_module.set_flat_param(self.decision_module.tunable_para_expert, False)
        self.planning_module.set_flat_param(self.planning_module.tunable_para_expert, False)
        self.control_module.set_flat_param(self.control_module.tunable_para_expert, False)

    def get_action(self, ego_state, surrounding_state):
        info = self.decision_module.get_action(ego_state, surrounding_state)
        cf_state, target_lane_num = info['cf_state'], info['target_lane_num']
        ref_points = self.planning_module.get_action(ego_state, cf_state, target_lane_num)
        action = self.control_module.get_action(ego_state, ref_points)
        info['target_lane_num'] = target_lane_num
        info['cf_state'] = cf_state
        return np.array([action[0], action[1], target_lane_num]), info