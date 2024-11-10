import copy
from typing import Tuple, List

import numpy as np
import traci
from gym.spaces import Box
from traci.exceptions import TraCIException

from multi_lane_env.env.single import MultiLaneSingleEnv
from multi_lane_env.vehicle.dynamics import BicycleVehicle


class MultiLaneFlowEnv(MultiLaneSingleEnv):
    def __init__(self, seed, label, config: dict = None) -> None:
        super(MultiLaneSingleEnv, self).__init__(seed, label, config)
        self.observation_space = Box(low=float('-inf'), high=float('inf'), shape=(int(self.config['max_surr_num'])*4+6,))
        self.action_space = Box(
            low=np.array([BicycleVehicle.MIN_STEER, BicycleVehicle.MIN_ACC], dtype=np.float32),
            high=np.array([BicycleVehicle.MAX_STEER, BicycleVehicle.MAX_ACC], dtype=np.float32),
            shape=(2,)
        )
        # initialize traffic flow
        self.conn = traci.getConnection(label)
        for i in range(int(self.config['init_flow'] / self.config['time_step'])):
            self.conn.simulationStep()
        self._add_vehicle('ego', 'ego')
        self.surr_vehicles = None

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            'sumo_dir': 'flow',
            'max_episode_steps': 1000,
            'sensor_range': 80,  # [m]
            'max_surr_num': 8,
            'init_flow': 100,  # [s]
            'reset_flow': 10,  # [s]
            'reset_min_gap': 40,  # [m]
            'track_start': 10,  # [s]
        })
        return config

    def _reset(self) -> None:
        # reset traffic flow
        for i in range(int(self.config['reset_flow'] / self.config['time_step'])):
            self.conn.simulationStep()

        self.target_lane = np.random.randint(self.config['lane_num'])
        self.depart_lane = np.random.randint(self.config['lane_num'])
        ego_x = None
        veh_ids = self.conn.lane.getLastStepVehicleIDs(f'0to1_{self.depart_lane}')
        last_x = 0
        for veh_id in veh_ids:
            x, y = self.conn.vehicle.getPosition(veh_id)
            if x - last_x >= self.config['reset_min_gap']:
                ego_x = (last_x + x) / 2 - BicycleVehicle.LENGTH / 2
                break
            last_x = x

        self.ego_vehicle = BicycleVehicle(
            position=[ego_x, self._lane_center(self.depart_lane) + np.random.uniform(-0.05, 0.05)],
            speed=np.random.uniform(*self.config['initial_speed_range']),
            heading=np.random.uniform(-0.01, 0.01),
            min_speed=self.config['min_speed'], max_speed=self.config['max_speed'])
        try:
            self._move_vehicle('ego', self.ego_vehicle)
        except TraCIException:
            self._add_vehicle('ego', 'ego')
            self._move_vehicle('ego', self.ego_vehicle)

        # add sensor
        self.conn.vehicle.subscribeContext(
            objectID='ego',
            domain=traci.constants.CMD_GET_VEHICLE_VARIABLE,
            dist=self.config['sensor_range'],
            varIDs=[
                traci.constants.VAR_POSITION,
                traci.constants.VAR_ANGLE,
                traci.constants.VAR_SPEED
            ]
        )

    def _get_obs(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ego_state, _, ego_obs = super()._get_obs()
        self.surr_vehicles = self.get_surrounding_vehicles()
        surr_state_noisy = copy.deepcopy(self.surr_vehicles)
        for surr in surr_state_noisy:
            noise = np.random.randn(4)
            surr[0] += 0.14 * noise[0]
            surr[1] += 0.14 * noise[1]
            surr[2] += 1.0 / 180. * np.pi * noise[2]
            surr[3] += 0.15 * noise[3]
        surr_state_noisy_output = np.array(surr_state_noisy)
        surr_state_noisy.sort(key=lambda v: (v[0] - self.ego_vehicle.position[0]) ** 2 +
                                              (v[1] - self.ego_vehicle.position[1]) ** 2)
        surr_num = min(len(surr_state_noisy), self.config['max_surr_num'])
        surr_obs = np.zeros((self.config['max_surr_num'], 4), dtype=np.float32)
        # normalize
        if surr_num > 0:
            surr_obs[:surr_num] = surr_state_noisy[:surr_num]
            surr_obs[:, 0] = (surr_obs[:, 0] - self.ego_vehicle.position[0]) / \
                            self.config['sensor_range']
            surr_obs[:, 1] = -surr_obs[:, 1] / (self.config['lane_num'] * self.config['lane_width'])
            surr_obs[:, 3] = surr_obs[:, 3] / self.config['max_speed']

        return ego_state, surr_state_noisy_output, np.concatenate((ego_obs, surr_obs.flatten()))

    # @staticmethod
    def get_surrounding_vehicles(self) -> List:
        sub_result = self.conn.vehicle.getContextSubscriptionResults('ego')
        surr_vehs = []
        for vid, info in sub_result.items():
            if vid != 'ego':
                x, y = info[traci.constants.VAR_POSITION]
                x -= BicycleVehicle.LENGTH / 2
                angle = info[traci.constants.VAR_ANGLE]
                phi = (90 - angle) * np.pi / 180
                u = info[traci.constants.VAR_SPEED]
                surr_vehs.append([x, y, phi, u])
        return surr_vehs

    def _get_cost_fail_info(self, action: np.ndarray) -> Tuple[float, bool, dict]:
        cost, fail, info = super()._get_cost_fail_info(action)
        fail = bool(self._check_collision() or fail)
        return cost, fail, info

    def _check_collision(self) -> bool:
        self_centers = self.ego_vehicle.circle_centers()

        surr_num = min(len(self.surr_vehicles), self.config['max_surr_num'])
        if surr_num > 0:
            surr_vehicles = np.array(self.surr_vehicles[:surr_num], dtype=np.float32)
            surr_x = surr_vehicles[:, 0]
            surr_y = surr_vehicles[:, 1]
            surr_phi = surr_vehicles[:, 2]
            d = (BicycleVehicle.LENGTH - BicycleVehicle.WIDTH) / 2
            surr_center_f = np.stack(
                ((surr_x + d * np.cos(surr_phi)), surr_y + d * np.sin(surr_phi)), axis=1)
            surr_center_r = np.stack(
                ((surr_x - d * np.cos(surr_phi)), surr_y - d * np.sin(surr_phi)), axis=1)
            surr_centers = np.stack((surr_center_f, surr_center_r), axis=1)

            self_centers_expand = np.tile(self_centers, (surr_vehicles.shape[0], 2, 1))
            surr_centers_expand = np.repeat(surr_centers, 2, axis=1)
            centers_dist = np.linalg.norm(surr_centers_expand - self_centers_expand, axis=-1)
            veh_dist = np.min(centers_dist, axis=-1) - BicycleVehicle.RADIUS * 2
            return (veh_dist < 0).any()
        else:
            False

    def _render(self, ax):
        super(MultiLaneFlowEnv, self)._render(ax)
        import matplotlib.patches as pc

        # draw surrounding vehicles
        veh_len = self.ego_vehicle.LENGTH
        veh_wid = self.ego_vehicle.WIDTH
        for sv in self.surr_vehicles:
            x, y, phi, _ = sv
            ax.add_patch(pc.Rectangle(
                (x - veh_len / 2, y - veh_wid / 2), veh_len, veh_wid, phi * 180 / np.pi,
                facecolor='w', edgecolor='k', zorder=1))


if __name__ == '__main__':
    env = MultiLaneFlowEnv()
    obs = env.reset()
    env.render()
    while True:
        action = np.array([0, 0], dtype=np.float32)
        # action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()
        else:
            obs = next_obs
