from typing import List

import numpy as np


class BicycleVehicle:
    LENGTH: float = 4.8  # [m]
    WIDTH: float = 2.0  # [m]
    LENGTH_FRONT: float = 1.06  # [m]
    LENGTH_REAR: float = 1.85  # [m]
    MASS: float = 1412  # [kg]
    INERTIA_Z: float = 1536.7  # [kg.m^2]
    STIFF_FRONT: float = -128916  # [N/rad]
    STIFF_REAR: float = -85944  # [N/rad]

    MIN_ACC: float = -5  # [m/s^2]
    MAX_ACC: float = 2  # [m/s^2]
    MIN_STEER: float = -0.3  # [rad]
    MAX_STEER: float = 0.3  # [rad]

    RADIUS: float = WIDTH * np.sqrt(2) / 2  # [m]

    def __init__(self, position: List[float], heading: float = 0, speed: float = 0,
                 min_speed: float = 1, max_speed: float = 40) -> None:
        self.position = position
        self.heading = heading
        self.speed = speed
        self.lateral_speed = 0
        self.yaw_rate = 0
        self.min_speed = min_speed
        self.max_speed = max_speed

    def step(self, action: np.ndarray, dt: float) -> None:
        acc = min(max(action[1], self.MIN_ACC), self.MAX_ACC)
        steer = min(max(action[0], self.MIN_STEER), self.MAX_STEER)

        phi = self.heading
        u = self.speed
        v = self.lateral_speed
        omega = self.yaw_rate
        lf = self.LENGTH_FRONT
        lr = self.LENGTH_REAR
        m = self.MASS
        Iz = self.INERTIA_Z
        kf = self.STIFF_FRONT
        kr = self.STIFF_REAR

        self.position[0] += dt * (u * np.cos(phi) - v * np.sin(phi))
        self.position[1] += dt * (v * np.cos(phi) + u * np.sin(phi))
        self.heading += dt * omega
        self.speed += dt * acc
        self.lateral_speed = (m * u * v + dt * (lf * kf - lr * kr) * omega -
                              dt * kf * steer * u - dt * (m * u ** 2 * omega)) / \
                             (m * u - dt * (kf + kr))
        self.yaw_rate = (Iz * u * omega + dt * (lf * kf - lr * kr) * v -
                         dt * lf * kf * steer * u) / \
                        (Iz * u - dt * (lf ** 2 * kf + lr ** 2 * kr))

        self.speed = min(max(self.speed, self.min_speed), self.max_speed)

    def circle_centers(self):
        x, y = self.position
        phi = self.heading
        d = (self.LENGTH - self.WIDTH) / 2
        return np.array([[x + d * np.cos(phi), y + d * np.sin(phi)],
                         [x - d * np.cos(phi), y - d * np.sin(phi)]])
