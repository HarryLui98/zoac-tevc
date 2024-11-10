from typing import Tuple

import numpy as np
from gym.spaces import Box

from multi_lane_env.env.base import BaseEnv
from multi_lane_env.vehicle.dynamics import BicycleVehicle


class MultiLaneSingleEnv(BaseEnv):
    def __init__(self, seed, label, config: dict = None) -> None:
        super(MultiLaneSingleEnv, self).__init__(seed, label, config)
        self.observation_space = Box(low=float('-inf'), high=float('inf'), shape=(6,))
        self.action_space = Box(
            low=np.array([BicycleVehicle.MIN_STEER, BicycleVehicle.MIN_ACC], dtype=np.float32),
            high=np.array([BicycleVehicle.MAX_STEER, BicycleVehicle.MAX_ACC], dtype=np.float32),
            shape=(2,)
        )
        self._add_vehicle('ego', 'ego')

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            'sumo_dir': 'single',
            'max_episode_steps': 1000,
            'track_start': 5,  # [s]
        })
        return config

    def _reset(self) -> None:
        self.target_lane = np.random.randint(self.config['lane_num'])
        self.depart_lane = np.random.randint(
            max(self.target_lane - 1, 0), min(self.target_lane + 2, self.config['lane_num']))
        self.planning_target_lane = self.depart_lane
        self.ego_vehicle = BicycleVehicle(
            position=[50, self._lane_center(self.depart_lane)],
            speed=np.random.uniform(*self.config['initial_speed_range']),
            min_speed=self.config['min_speed'], max_speed=self.config['max_speed'])
        self._move_vehicle('ego', self.ego_vehicle)

    def _step(self, action: np.ndarray) -> None:
        self.ego_vehicle.step(action, dt=self.config['time_step'])
        self._move_vehicle('ego', self.ego_vehicle)

    def _get_obs(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        road_width = self.config['lane_num'] * self.config['lane_width']
        obs = np.array([
            -self.ego_vehicle.position[1] / road_width,
            self.ego_vehicle.heading / self.config['max_heading'],
            self.ego_vehicle.speed / self.config['max_speed'],
            self.ego_vehicle.lateral_speed,
            self.ego_vehicle.yaw_rate,
            -self.target_y / road_width
        ], dtype=np.float32)
        ego_state = np.array([
            self.ego_vehicle.position[0],
            self.ego_vehicle.speed,
            self.ego_vehicle.lateral_speed,
            self.ego_vehicle.yaw_rate,
            self.ego_vehicle.position[1],
            self.ego_vehicle.heading
        ], dtype=np.float32)
        return ego_state, np.zeros([1, 4]), obs

    def _get_cost_fail_info(self, action: np.ndarray) -> Tuple[float, bool, dict]:
        speed_cost = -0.1 * (self.ego_vehicle.speed - self.config['target_speed']) ** 2
        smooth_cost = -1.0 * action[1] ** 2 - 60. * action[0] ** 2 - 300. * self.ego_vehicle.heading ** 2 \
                      - 30. * self.ego_vehicle.lateral_speed ** 2 - 300. * self.ego_vehicle.yaw_rate ** 2
        rule_cost = -0.6 * (self.ego_vehicle.position[1] - self.target_y) ** 2
        speed_kmh = self.ego_vehicle.speed * 3.6
        resistance = 1.2256 / 25.92 * 0.34 * 1.0 * 0.233 * speed_kmh ** 2 \
                        + 9.8 * 1412. * 1.75 / 1000. * (0.0328 * speed_kmh + 4.575)
        power = (resistance + 1412. * action[1] * 1.04) / (3600. * 0.92) * speed_kmh
        alpha_0, alpha_1, alpha_2 = 4.025e-4, 7.2216e-5, 1e-6
        fuel_cost = -1000 * (alpha_0 + alpha_1 * power + alpha_2 * power ** 2) if power >= 0 else -1000 * alpha_0
        fail = bool(self._check_out_of_lane()
                    or abs(self.ego_vehicle.heading) > self.config['max_heading']
                    or abs(self.ego_vehicle.position[1] - self.target_y) >
                    self.config['max_pos_error'])
        cost = 7. + speed_cost + smooth_cost + rule_cost + fuel_cost
        info = {
            'cost/speed': speed_cost,
            'cost/smooth': smooth_cost,
            'cost/rule': rule_cost,
            'cost/fuel': fuel_cost,
        }
        return cost, fail, info

    def _check_out_of_lane(self) -> bool:
        circle_centers = self.ego_vehicle.circle_centers()
        y_min = -self.config['lane_num'] * self.config['lane_width'] + BicycleVehicle.RADIUS
        y_max = -BicycleVehicle.RADIUS
        return bool(circle_centers[0, 1] < y_min
                    or circle_centers[0, 1] > y_max
                    or circle_centers[1, 1] < y_min
                    or circle_centers[1, 1] > y_max)

    def _lane_center(self, lane: int) -> float:
        return (-self.config['lane_num'] + lane + 0.5) * self.config['lane_width']

    @property
    def target_y(self) -> float:
        return self._lane_center(self.target_lane)

    def _render(self, ax):
        import matplotlib.patches as pc

        # draw road
        for i in range(self.config['lane_num'] + 1):
            if i == 0 or i == self.config['lane_num']:
                line_type = '-'
            else:
                line_type = '--'
            y = -i * self.config['lane_width']
            ax.plot([0, self.config['lane_length']], [y] * 2, 'k' + line_type, lw=1, zorder=0)

        # draw ego vehicle
        ego_x, ego_y = self.ego_vehicle.position
        veh_len = self.ego_vehicle.LENGTH
        veh_wid = self.ego_vehicle.WIDTH
        phi = self.ego_vehicle.heading
        ax.add_patch(pc.Rectangle(
            (ego_x - veh_len / 2, ego_y - veh_wid / 2), veh_len, veh_wid, phi * 180 / np.pi,
            facecolor='w', edgecolor='r', zorder=1))

        # draw reference paths
        for i in range(0, self.config['lane_num']):
            y = self._lane_center(i)
            if y == self._lane_center(self.planning_target_lane):
                alpha = 1
            else:
                alpha = 0.3
            ax.plot([0, self.config['lane_length']], [y] * 2, 'b', lw=1, alpha=alpha, zorder=0)

        # draw texts
        left_x = ego_x - 70
        mid_y = -self.config['lane_num'] / 2 * self.config['lane_width']
        top_y = mid_y + 30
        delta_y = 5
        ego_speed = self.ego_vehicle.speed * 3.6  # [km/h]
        episode_time = self._episode_step * self.config['time_step']
        ax.text(left_x, top_y, f'time: {episode_time:.1f}s')
        ax.text(left_x, top_y - delta_y, f'speed: {ego_speed:.1f}km/h')


if __name__ == '__main__':
    env = MultiLaneSingleEnv()
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
