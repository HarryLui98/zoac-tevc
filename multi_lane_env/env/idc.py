import numpy as np

from multi_lane_env import MultiLaneFlowEnv


class IDCMultiLaneFlowEnv(MultiLaneFlowEnv):
    def __init__(self, seed, label, config: dict = None):
        super().__init__(seed, label, config)
        self.planning_target_lane = None

    def _reset(self) -> None:
        super(IDCMultiLaneFlowEnv, self)._reset()
        self.target_lane = self.depart_lane
        self.planning_target_lane = self.depart_lane

    @property
    def target_y(self) -> float:
        curr_lane = int(self.ego_vehicle.position[1] / self.config['lane_width'] +
                        self.config['lane_num'])
        self.target_lane = curr_lane
        return self._lane_center(self.target_lane)

    def get_reference_obs(self) -> np.ndarray:
        org_target_lane = self.target_lane
        ref_obs = []
        for i in range(self.config['lane_num']):
            self.target_lane = i
            obs = super(IDCMultiLaneFlowEnv, self)._get_obs()
            ref_obs.append(obs)
        self.target_lane = org_target_lane
        return np.stack(ref_obs)
