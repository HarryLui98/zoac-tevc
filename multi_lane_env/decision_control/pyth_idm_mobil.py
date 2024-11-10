import random
import math
import numpy as np
import gym
from multi_lane_env.vehicle.dynamics import BicycleVehicle


class BehaviorDecision_IDM_MOBIL(object):
    def __init__(self, seed, config):
        random.seed(seed)
        np.random.seed(seed)
        # Tunable parameters (8):
        # idm - T, s0, delta, a, b
        # mobil - p, ath, bsafe
        self.tunable_para_high = np.array([3., 3., 10., 2., 5., 1.0, 0.5, 5.]) # T 1-3 s0 4-6
        self.tunable_para_low = np.array([1., 1., 1., 0.5, 0.5, 0., 0.05, 0.5])
        self.lin_para_gain = 0.5 * (self.tunable_para_high - self.tunable_para_low)
        self.lin_para_bias = 0.5 * (self.tunable_para_high + self.tunable_para_low)
        # self.tunable_para_mapped = np.random.uniform(-1, 1, 8)
        self.tunable_para_mapped = np.zeros(8)
        self.tunable_para_unmapped = self.tunable_para_transform(self.tunable_para_mapped, after_map=True)
        self.tunable_para_sigma = 0.01 * np.ones(8)
        self.tunable_para_expert = np.array([2.0, 2.0, 4., 1.5, 2.0, 0.2, 0.1, 4.])
        self.tunable_para_expert_mapped = self.tunable_para_transform(self.tunable_para_expert, after_map=False)
        self.para_num = 8
        self.gamma = 0.99
        self.config = config
        self.step_T = 0.1
        self.info = {'lk_time': 0., 'target_lane_num': None, 'current_mode': 'CC', 'cf_state': None}

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

    def reset(self):
        self.info = {'lk_time': 0., 'target_lane_num': None, 'current_mode': 'CC', 'cf_state': None}

    def get_action(self, ego_state, surrounding_state):
        # ego_state: x, u, v, yaw, y, phi
        # surrounding_state: x, y, phi, u
        ego_lane = int(ego_state[4] / self.config['lane_width'] + self.config['lane_num'])
        if self.info['target_lane_num'] is None:
            self.info['target_lane_num'] = ego_lane
        x_rel_lfmin, x_rel_lrmax = self.config['sensor_range'], -self.config['sensor_range']
        lf_state = [ego_state[0] + self.config['sensor_range'],
                    (ego_lane + 1 - self.config['lane_num'] + 0.5) * self.config['lane_width'],
                    0., self.config['target_speed']]
        lr_state = [ego_state[0] - self.config['sensor_range'],
                    (ego_lane + 1 - self.config['lane_num'] + 0.5) * self.config['lane_width'],
                    0., self.config['target_speed']]
        x_rel_cfmin, x_rel_crmax = self.config['sensor_range'], -self.config['sensor_range']
        cf_state = [ego_state[0] + self.config['sensor_range'],
                    (ego_lane - self.config['lane_num'] + 0.5) * self.config['lane_width'],
                    0., self.config['target_speed']]
        cr_state = [ego_state[0] - self.config['sensor_range'],
                    (ego_lane - self.config['lane_num'] + 0.5) * self.config['lane_width'],
                    0., self.config['target_speed']]
        x_rel_rfmin, x_rel_rrmax = self.config['sensor_range'], -self.config['sensor_range']
        rf_state = [ego_state[0] + self.config['sensor_range'],
                    (ego_lane - 1 - self.config['lane_num'] + 0.5) * self.config['lane_width'],
                    0., self.config['target_speed']]
        rr_state = [ego_state[0] - self.config['sensor_range'],
                    (ego_lane - 1 - self.config['lane_num'] + 0.5) * self.config['lane_width'],
                    0., self.config['target_speed']]
        llc_flag, rlc_flag = True, True
        for veh_idx in range(len(surrounding_state)):
            veh_lane = int(surrounding_state[veh_idx][1] / self.config['lane_width'] + self.config['lane_num'])
            veh_x_relative = surrounding_state[veh_idx][0] - ego_state[0]
            if veh_lane == ego_lane + 1:
                if -BicycleVehicle.LENGTH <= veh_x_relative <= BicycleVehicle.LENGTH:
                    llc_flag = False
                if 0 < veh_x_relative < x_rel_lfmin:
                    lf_idx = veh_idx
                    lf_state = surrounding_state[lf_idx]
                    x_rel_lfmin = veh_x_relative
                elif x_rel_lrmax < veh_x_relative <= 0:
                    lr_idx = veh_idx
                    lr_state = surrounding_state[lr_idx]
                    x_rel_lrmax = veh_x_relative
            if veh_lane == ego_lane:
                if 0 < veh_x_relative < x_rel_cfmin:
                    cf_idx = veh_idx
                    cf_state = surrounding_state[cf_idx]
                    x_rel_cfmin = veh_x_relative
                elif x_rel_crmax < veh_x_relative <= 0:
                    cr_idx = veh_idx
                    cr_state = surrounding_state[cr_idx]
                    x_rel_crmax = veh_x_relative
            if veh_lane == ego_lane - 1:
                if -BicycleVehicle.LENGTH <= veh_x_relative <= BicycleVehicle.LENGTH:
                    rlc_flag = False
                if 0 < veh_x_relative < x_rel_rfmin:
                    rf_idx = veh_idx
                    rf_state = surrounding_state[rf_idx]
                    x_rel_rfmin = veh_x_relative
                elif x_rel_rrmax < veh_x_relative <= 0:
                    rr_idx = veh_idx
                    rr_state = surrounding_state[rr_idx]
                    x_rel_rrmax = veh_x_relative
        p = self.tunable_para_unmapped[5]
        ath = self.tunable_para_unmapped[6]
        bsafe = self.tunable_para_unmapped[7]
        self.info['cf_state'] = cf_state
        if self.info['current_mode'] == 'LC':
            self.info['lk_time'] = 0.
            ego_y = ego_state[4]
            target_y = (-self.config['lane_num'] + self.info['target_lane_num'] + 0.5) * self.config['lane_width']
            if abs(ego_y - target_y) < 0.1:
                if cf_state[0] - ego_state[0] >= 79.9:
                    self.info['current_mode'] = 'CC'
                else:
                    self.info['current_mode'] = 'ACC'
        else:
            self.info['lk_time'] += self.step_T
            if cf_state[0] - ego_state[0] >= 79.9:
                self.info['current_mode'] = 'CC'
            else:
                # if change left
                if llc_flag:
                    l_ego_new_acc = self.get_longitudinal_acc([ego_state[0], ego_state[1] * math.cos(ego_state[5])],
                                                            [lf_state[0], lf_state[3] * math.cos(lf_state[2])])
                    l_ego_old_acc = self.get_longitudinal_acc([ego_state[0], ego_state[1] * math.cos(ego_state[5])],
                                                            [cf_state[0], cf_state[3] * math.cos(cf_state[2])])
                    l_newfollower_new_acc = self.get_longitudinal_acc([lr_state[0], lr_state[3] * math.cos(lr_state[2])],
                                                                    [ego_state[0], ego_state[1] * math.cos(ego_state[5])])
                    l_newfollower_old_acc = self.get_longitudinal_acc([lr_state[0], lr_state[3] * math.cos(lr_state[2])],
                                                                    [lf_state[0], lf_state[3] * math.cos(lf_state[2])])
                    l_oldfollower_new_acc = self.get_longitudinal_acc([cr_state[0], cr_state[3] * math.cos(cr_state[2])],
                                                                    [cf_state[0], cf_state[3] * math.cos(cf_state[2])])
                    l_oldfollower_old_acc = self.get_longitudinal_acc([cr_state[0], cr_state[3] * math.cos(cr_state[2])],
                                                                    [ego_state[0], ego_state[1] * math.cos(ego_state[5])])
                    left_change = l_ego_new_acc - l_ego_old_acc + p * (l_newfollower_new_acc - l_newfollower_old_acc +
                                                                l_oldfollower_new_acc - l_oldfollower_old_acc)
                else:
                    left_change = -100
                # if change right
                if rlc_flag:
                    r_ego_new_acc = self.get_longitudinal_acc([ego_state[0], ego_state[1] * math.cos(ego_state[5])],
                                                            [rf_state[0], rf_state[3] * math.cos(rf_state[2])])
                    r_ego_old_acc = self.get_longitudinal_acc([ego_state[0], ego_state[1] * math.cos(ego_state[5])],
                                                            [cf_state[0], cf_state[3] * math.cos(cf_state[2])])
                    r_newfollower_new_acc = self.get_longitudinal_acc([rr_state[0], rr_state[3] * math.cos(rr_state[2])],
                                                                    [ego_state[0], ego_state[1] * math.cos(ego_state[5])])
                    r_newfollower_old_acc = self.get_longitudinal_acc([rr_state[0], rr_state[3] * math.cos(rr_state[2])],
                                                                    [rf_state[0], rf_state[3] * math.cos(rf_state[2])])
                    r_oldfollower_new_acc = self.get_longitudinal_acc([cr_state[0], cr_state[3] * math.cos(cr_state[2])],
                                                                    [cf_state[0], cf_state[3] * math.cos(cf_state[2])])
                    r_oldfollower_old_acc = self.get_longitudinal_acc([cr_state[0], cr_state[3] * math.cos(cr_state[2])],
                                                                    [ego_state[0], ego_state[1] * math.cos(ego_state[5])])
                    right_change = r_ego_new_acc - r_ego_old_acc + p * (r_newfollower_new_acc - r_newfollower_old_acc +
                                                                r_oldfollower_new_acc - r_oldfollower_old_acc)
                else:
                    right_change = -100
                if ego_lane == 0:
                    if left_change > ath and l_newfollower_new_acc > -bsafe and self.info['lk_time'] >= 3.0:
                        self.info['target_lane_num'] = ego_lane + 1
                        self.info['current_mode'] = 'LC'
                    else:
                        self.info['target_lane_num'] = ego_lane
                        self.info['current_mode'] = 'ACC'
                elif ego_lane == self.config['lane_num'] - 1:
                    if right_change > ath and r_newfollower_new_acc > -bsafe and self.info['lk_time'] >= 3.0:
                        self.info['target_lane_num'] = ego_lane - 1
                        self.info['current_mode'] = 'LC'
                    else:
                        self.info['target_lane_num'] = ego_lane
                        self.info['current_mode'] = 'ACC'
                else:
                    if (left_change > ath and l_newfollower_new_acc > -bsafe and self.info['lk_time'] >= 3.0) \
                        or (right_change > ath and r_newfollower_new_acc > -bsafe and self.info['lk_time'] >= 3.0):
                        if left_change > right_change:
                            self.info['target_lane_num'] = ego_lane + 1
                            self.info['current_mode'] = 'LC'
                        else:
                            self.info['target_lane_num'] = ego_lane - 1
                            self.info['current_mode'] = 'LC'
                    else:
                        self.info['target_lane_num'] = ego_lane
                        self.info['current_mode'] = 'ACC'
        return self.info



