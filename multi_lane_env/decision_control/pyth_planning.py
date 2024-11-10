import random
import math
import numpy as np
import gym


class Planning_IDC(object):
    def __init__(self, seed, config):
        random.seed(seed)
        np.random.seed(seed)
        self.Np = 25
        self.step_T = 0.1
        # Tunable parameters (8):
        # idm - T(0), s0(1), delta(2), a(3), b(4)
        # mobil - p(5), ath(6), bsafe(7)
        self.tunable_para_high = np.array([3., 3., 10., 2., 5., 1.0, 0.5, 5.])
        self.tunable_para_low = np.array([1., 1., 1., 0.5, 0.5, 0., 0.05, 0.5])
        self.lin_para_gain = 0.5 * (self.tunable_para_high - self.tunable_para_low)
        self.lin_para_bias = 0.5 * (self.tunable_para_high + self.tunable_para_low)
        self.tunable_para_mapped = np.zeros(8)
        self.tunable_para_unmapped = self.tunable_para_transform(self.tunable_para_mapped, after_map=True)
        self.tunable_para_sigma = 0.01 * np.ones(8)
        self.tunable_para_expert = np.array([2.0, 2.0, 4., 1.5, 2.0, 0.5, 0.1, 4.])
        self.tunable_para_expert_mapped = self.tunable_para_transform(self.tunable_para_expert, after_map=False)
        self.para_num = 8
        self.gamma = 0.99
        self.config = config

    def tunable_para_transform(self, para_in, after_map):
        if after_map:
            lin_para_mapped = para_in
            lin_para_unmapped = self.lin_para_gain * lin_para_mapped + self.lin_para_bias
            para_out = lin_para_unmapped
        else:
            lin_para_unmapped = para_in
            lin_para_mapped = (lin_para_unmapped - self.lin_para_bias) / self.lin_para_gain
            para_out = lin_para_mapped
        return para_out

    def get_flat_param(self, after_map=True):
        if after_map:
            return self.tunable_para_mapped
        else:
            return self.tunable_para_unmapped

    def set_flat_param(self, para, after_map=True):
        if after_map:
            para = np.clip(para, -1., 1.)
            self.tunable_para_mapped = para
            para_unmapped = self.tunable_para_transform(para, after_map)
            self.tunable_para_unmapped = para_unmapped
        else:
            para = np.clip(para, self.tunable_para_low, self.tunable_para_high)
            self.tunable_para_unmapped = para
            para_mapped = self.tunable_para_transform(para, after_map)
            self.tunable_para_mapped = para_mapped

    def get_longitudinal_acc(self, state, leading_state):
        # input format: x, u
        v = max(state[1], 1e-2)
        delta_v = state[1] - leading_state[1]
        s = leading_state[0] - state[0] - 4.8
        T, s0, delta = self.tunable_para_unmapped[0], self.tunable_para_unmapped[1], self.tunable_para_unmapped[2]
        a, b = self.tunable_para_unmapped[3], self.tunable_para_unmapped[4]
        s_star = s0 + v * T + v * delta_v / (2 * math.sqrt(a * b))
        acc = a * (1 - (v / self.config['target_speed']) ** delta - (s_star / (s + 1e-2)) ** 2)
        acc = np.clip(acc, -5., 2.)
        return acc

    def get_action(self, ego_state, cf_state, target_lane_num):
        # initial_state: x, u, v, yaw, y, phi
        # cf_state: x, y, phi, u
        # reference_point: x, u, v, yaw, y, phi
        ref_points = []
        leading_x, leading_u = cf_state[0], cf_state[3] * math.cos(cf_state[2])
        ego_x, ego_u = ego_state[0], ego_state[1] * math.cos(ego_state[5])
        if leading_x - ego_x < 79.9:
            for i in range(self.Np):
                v, phi, yaw = 0., 0., 0.
                target_y = (-self.config['lane_num'] + target_lane_num + 0.5) * self.config['lane_width']
                ego_acc = self.get_longitudinal_acc([ego_x, ego_u], [leading_x, leading_u])
                ego_x += ego_u * self.step_T
                leading_x += leading_u * self.step_T
                ego_u += ego_acc * self.step_T
                ref_points.append(ego_x)
                ref_points.append(ego_u)
                ref_points.append(v)
                ref_points.append(yaw)
                ref_points.append(target_y)
                ref_points.append(phi)
        else:
            for i in range(self.Np):
                v, phi, yaw = 0., 0., 0.
                target_y = (-self.config['lane_num'] + target_lane_num + 0.5) * self.config['lane_width']
                ego_acc = 0.
                ego_x += self.config['target_speed'] * self.step_T
                ego_u = self.config['target_speed']
                ref_points.append(ego_x)
                ref_points.append(ego_u)
                ref_points.append(v)
                ref_points.append(yaw)
                ref_points.append(target_y)
                ref_points.append(phi)
        return np.array(ref_points).astype(np.float32)
