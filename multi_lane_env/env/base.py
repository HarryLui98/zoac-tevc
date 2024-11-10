import os
from typing import Tuple

import gym
import numpy as np
import traci


class BaseEnv(gym.Env):
    def __init__(self, seed=100, label='eval', config: dict = None) -> None:
        np.random.seed(seed)
        self.config = self.default_config()
        self.configure(config)
        self.ego_vehicle = None
        self.depart_lane = None
        self.target_lane = None
        self.planning_target_lane = None
        self._episode_step = None
        self.label = label
        self.conn = self._start_sumo(label)

    @classmethod
    def default_config(cls) -> dict:
        return {
            'lane_num': 4,
            'lane_width': 3.75,  # [m]
            'lane_length': 2000,  # [m]
            'initial_speed_range': [40 / 3.6, 50 / 3.6],  # [m/s]
            'target_speed': 50 / 3.6,  # [m/s]
            'min_speed': 20 / 3.6,  # [m/s]
            'max_speed': 80 / 3.6,  # [m/s]
            'max_heading': np.pi / 4,  # [rad]
            'max_pos_error': 2 * 3.75,  # [m]
            'time_step': 0.1,  # [s]
            'max_episode_steps': 1000,
            'sumo_dir': None,
            'sumo_label': 'default'
        }

    def configure(self, config: dict) -> None:
        if config is not None:
            self.config.update(config)

    def reset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._reset()
        self.conn.simulationStep()
        self._episode_step = 0
        return self._get_obs()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, bool, dict]:
        self._step(action)
        self.conn.simulationStep()
        self._episode_step += 1
        # self.planning_target_lane = int(action[-1])
        ego_state, surrounding_state, obs = self._get_obs()
        cost, fail, info = self._get_cost_fail_info(action)
        reward = cost - fail * 2000
        if self._episode_step >= self.config['max_episode_steps']:
            info['TimeLimit.truncated'] = not fail
            done = True
        else:
            done = fail
        info['dead'] = fail
        info['ego_state'] = ego_state
        info['surr_state'] = surrounding_state
        return obs, reward, done, info

    def _start_sumo(self, label):
        par_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
        sumo_config_path = par_dir + f'/sumo_files/{self.config["sumo_dir"]}/sumocfg'
        sumo_cmd = ['sumo', '-c', sumo_config_path,
                    '--step-length', str(self.config['time_step'])]
        traci.start(sumo_cmd, label=label)
        conn = traci.getConnection(self.label)
        return conn

    def _reset(self) -> None:
        raise NotImplementedError

    def _step(self, action: np.ndarray) -> None:
        raise NotImplementedError

    def _get_obs(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError

    def _get_cost_fail_info(self, action: np.ndarray) -> Tuple[float, bool, dict]:
        raise NotImplementedError

    # @staticmethod
    def _add_vehicle(self, vehID, typeID):
        self.conn.vehicle.add(vehID=vehID, routeID='0', typeID=typeID)
        self.conn.vehicle.setSpeedMode(vehID, 0)  # strictly follow speed control commands
        self.conn.vehicle.setLaneChangeMode(vehID, 0)  # disable auto lane change

    # @staticmethod
    def _move_vehicle(self, vehID, veh):
        x_head = veh.position[0] + veh.LENGTH / 2 * np.cos(veh.heading)
        y_head = veh.position[1] + veh.LENGTH / 2 * np.sin(veh.heading)
        angle = 90 - veh.heading * 180 / np.pi
        self.conn.vehicle.moveToXY(vehID=vehID, edgeID='', lane=-1, x=x_head, y=y_head,
                               angle=angle, keepRoute=2)

    def render(self, mode='human'):
        import matplotlib.pyplot as plt

        plt.figure(num=0, figsize=(6.4, 3.2))
        plt.clf()  # clear the current figure before rendering the new frame
        ego_x, ego_y = self.ego_vehicle.position
        mid_y = -self.config['lane_num'] / 2 * self.config['lane_width']
        ax = plt.axes(xlim=(ego_x - 80, ego_x + 80), ylim=(mid_y - 10, mid_y + 10))
        ax.set_aspect('equal')
        plt.axis('off')

        self._render(ax)

        plt.tight_layout()
        plt.pause(0.01)

    def _render(self, ax):
        raise NotImplementedError

    def close(self):
        self.conn.close()
