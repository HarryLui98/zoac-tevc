import random
import math
import numpy as np
import casadi
import gym
import copy


class ModelPredictiveController(object):
    def __init__(self, seed, config):
        random.seed(seed)
        np.random.seed(seed)
        self.Np = 25
        self.step_T = 0.1
        self.action_space = gym.spaces.Box(
            low=np.array([-0.3, -5., -np.inf], dtype=np.float32),
            high=np.array([0.3, 2., np.inf], dtype=np.float32),
            shape=(3,)
        )
        # Tunable parameters (19):
        # model - Cf(0), Cr(1), a(2), b(3), m(4), Iz(5)
        # stage cost - x_w(6), y_w(7), phi_w(8), v_w(9), yaw_w(10), str_w(17), acc_w(18) (u_w is set as 0.1) - log space
        # terminal cost - x_w(11), y_w(12), phi_w(13), v_w(14), yaw_w(15), u_w(16) - log space
        self.tunable_para_high = np.array([-8e4, -8e4, 2.2, 2.2, 2000, 2000,
                                           1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3,
                                           1e3, 1e3, 1e3, 1e3, 1e3, 1e3])
        self.tunable_para_low = np.array([-16e4, -16e4, 0.8, 0.8, 1000, 1000,
                                          1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5,
                                          1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5])
        self.x0 = ([0, 0, 0, 0, 0, 0, 0, 0] * (self.Np + 1))
        self.x0.pop(-1)
        self.x0.pop(-1)
        self.ref_p = None
        self.surr_veh = None
        self.lin_para_gain = 0.5 * (self.tunable_para_high[:6] - self.tunable_para_low[:6])
        self.lin_para_bias = 0.5 * (self.tunable_para_high[:6] + self.tunable_para_low[:6])
        self.log_para_gain = 0.5 * (np.log10(self.tunable_para_high[6:]) - np.log10(self.tunable_para_low[6:]))
        self.log_para_bias = 0.5 * (np.log10(self.tunable_para_high[6:]) + np.log10(self.tunable_para_low[6:]))
        self.tunable_para_mapped = np.zeros(19)
        self.tunable_para_unmapped = self.tunable_para_transform(self.tunable_para_mapped, after_map=True)
        self.tunable_para_expert = np.array([-128916, -85944, 1.06, 1.85, 1412, 1536.7,
                                             1e-5, 0.6, 300., 30., 300.,
                                             1e-5, 0.6, 300., 30., 300., 0.1, 60., 1.0])
        self.tunable_para_expert_mapped = self.tunable_para_transform(self.tunable_para_expert, after_map=False)
        self.para_num = 19
        self.model_para_num = 6
        self.gamma = 0.99
        self.config = config

    def tunable_para_transform(self, para_in, after_map):
        if after_map:
            lin_para_mapped = para_in[:6]
            log_para_mapped = para_in[6:]
            lin_para_unmapped = self.lin_para_gain * lin_para_mapped + self.lin_para_bias
            log_para_unmapped = np.power(10, self.log_para_gain * log_para_mapped + self.log_para_bias)
            para_out = np.concatenate((lin_para_unmapped, log_para_unmapped))
        else:
            lin_para_unmapped = para_in[:6]
            log_para_unmapped = para_in[6:]
            lin_para_mapped = (lin_para_unmapped - self.lin_para_bias) / self.lin_para_gain
            log_para_mapped = (np.log10(log_para_unmapped) - self.log_para_bias) / self.log_para_gain
            para_out = np.concatenate((lin_para_mapped, log_para_mapped))
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

    def step_forward(self, state, action):
        x, v_x, v_y, r, y, phi = state[0], state[1], state[2], state[3], state[4], state[5]
        steer, a_x = action[0], action[1]
        C_f, C_r, a, b, mass, I_z = self.tunable_para_unmapped[:6].tolist()
        tau = self.step_T
        next_state = [x + tau * (v_x * casadi.cos(phi) - v_y * casadi.sin(phi)),
                      v_x + tau * a_x,
                      (mass * v_y * v_x + tau * (a * C_f - b * C_r) * r -
                       tau * C_f * steer * v_x - tau * mass * casadi.power(v_x, 2) * r)
                      / (mass * v_x - tau * (C_f + C_r)),
                      (I_z * r * v_x + tau * (a * C_f - b * C_r) * v_y
                       - tau * a * C_f * steer * v_x) /
                      (I_z * v_x - tau * (casadi.power(a, 2) * C_f + casadi.power(b, 2) * C_r)),
                      y + tau * (v_x * casadi.sin(phi) + v_y * casadi.cos(phi)),
                      phi + tau * r]
        return next_state
    
    def reset(self, first=True):
        x = casadi.SX.sym('x', 6)
        u = casadi.SX.sym('u', 2)
        self.w = []
        self.ref_list = []
        self.G = []
        self.J = 0
        Xk = casadi.SX.sym('X0', 6)
        self.w += [Xk]
        for k in range(1, self.Np + 1):
            f = casadi.vertcat(*self.step_forward(x, u))
            F = casadi.Function("F", [x, u], [f])
            Uname = 'U' + str(k - 1)
            Uk = casadi.SX.sym(Uname, 2)
            self.w += [Uk]
            Fk = F(Xk, Uk)
            Xname = 'X' + str(k)
            Xk = casadi.SX.sym(Xname, 6)
            self.w += [Xk]
            # dynamic_state: x, y, u, v, phi, yaw
            self.G += [Fk - Xk]
            REFname = 'REF' + str(k)
            REFk = casadi.SX.sym(REFname, 6)
            self.ref_list += [REFk]
            if k < self.Np:
                ref_cost = 0.1 * casadi.power(self.w[k * 2][1] - self.ref_list[k - 1][1], 2)  # u
                ref_cost += self.tunable_para_unmapped[6] * casadi.power(self.w[k * 2][0] - self.ref_list[k - 1][0], 2)  # x
                ref_cost += self.tunable_para_unmapped[7] * casadi.power(self.w[k * 2][4] - self.ref_list[k - 1][4], 2)  # y
                ref_cost += self.tunable_para_unmapped[8] * casadi.power(self.w[k * 2][5] - self.ref_list[k - 1][5], 2)  # phi
                ref_cost += self.tunable_para_unmapped[9] * casadi.power(self.w[k * 2][2] - self.ref_list[k - 1][2], 2)  # v
                ref_cost += self.tunable_para_unmapped[10] * casadi.power(self.w[k * 2][3] - self.ref_list[k - 1][3], 2)  # yaw
                ref_cost *= casadi.power(self.gamma, k)
            else:
                ref_cost = self.tunable_para_unmapped[16] * casadi.power(self.w[k * 2][1] - self.ref_list[k - 1][1], 2)  # u
                ref_cost += self.tunable_para_unmapped[11] * casadi.power(self.w[k * 2][0] - self.ref_list[k - 1][0], 2)  # x
                ref_cost += self.tunable_para_unmapped[12] * casadi.power(self.w[k * 2][4] - self.ref_list[k - 1][4], 2)  # y
                ref_cost += self.tunable_para_unmapped[13] * casadi.power(self.w[k * 2][5] - self.ref_list[k - 1][5], 2)  # phi
                ref_cost += self.tunable_para_unmapped[14] * casadi.power(self.w[k * 2][2] - self.ref_list[k - 1][2], 2)  # v
                ref_cost += self.tunable_para_unmapped[15] * casadi.power(self.w[k * 2][3] - self.ref_list[k - 1][3], 2)  # yaw
                ref_cost *= casadi.power(self.gamma, k)
            act_cost = self.tunable_para_unmapped[17] * casadi.power(self.w[k * 2 - 1][0], 2)  # steer
            act_cost += self.tunable_para_unmapped[18] * casadi.power(self.w[k * 2 - 1][1], 2)  # ax
            act_cost *= casadi.power(self.gamma, k - 1)
            self.J += (ref_cost + act_cost)
        nlp = dict(f=self.J, g=casadi.vertcat(*self.G), x=casadi.vertcat(*self.w), p=casadi.vertcat(*self.ref_list))
        # self.S = casadi.nlpsol('S', 'ipopt', nlp)
        self.S = casadi.nlpsol('S', 'ipopt', nlp, {'ipopt.max_iter': 200, 'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0})
        if first:
            self.x0 = ([0, 0, 0, 0, 0, 0, 0, 0] * (self.Np + 1))
            self.x0.pop(-1)
            self.x0.pop(-1)

    def get_action(self, initial_state, reference_state):
        # initial_state: x, u, v, yaw, y, phi
        # reference_state: x, u, v, yaw, y, phi * Np
        self.ref_p = []
        lbw = []
        ubw = []
        lbg = []
        ubg = []
        lbw += initial_state.tolist()
        ubw += initial_state.tolist()
        for k in range(1, self.Np + 1):
            lbw += [self.action_space.low[0], self.action_space.low[1]]
            ubw += [self.action_space.high[0], self.action_space.high[1]]
            ubw += [casadi.inf, 80. / 3.6, casadi.inf, casadi.inf, -1.4, casadi.pi / 4]
            lbw += [-casadi.inf, 20. / 3.6, -casadi.inf, -casadi.inf, -13.6, -casadi.pi / 4]
            ubg += [0., 0., 0., 0., 0., 0.]
            lbg += [0., 0., 0., 0., 0., 0.]
            self.ref_p += reference_state[(k - 1) * 6: k * 6].tolist()
        r = self.S(lbx=casadi.vertcat(*lbw), ubx=casadi.vertcat(*ubw), x0=self.x0, lbg=casadi.vertcat(*lbg), ubg=casadi.vertcat(*ubg), p=self.ref_p)
        X = np.array(r['x']).tolist()
        action = np.array([X[6][0], X[7][0]])
        self.x0 = casadi.DM(
            X[8:] + X[-8] + X[-7] + X[-6] + X[-5] + X[-4] + X[-3] + X[-2] + X[-1])  # for faster optimization
        return action
