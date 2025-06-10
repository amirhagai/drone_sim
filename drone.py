import numpy as np
from typing import Optional, Tuple

class Drone:
    def __init__(self, position: np.ndarray, velocity: np.ndarray, max_accel: float, max_heading_change: float):
        self.position = position  # NED coordinates (3D)
        self.velocity = velocity  # 3D velocity
        self.max_accel = max_accel
        self.max_heading_change = max_heading_change  # radians per step
        self.last_arrived_point = position.copy()

    def step(self, dt: float, waypoint: Optional[np.ndarray]):
        """
        Move the drone towards the waypoint, respecting max heading change and acceleration.
        """
        if waypoint is not None:
            direction = waypoint - self.position
            direction_norm = np.linalg.norm(direction)
            if direction_norm == 0:
                return
            desired_heading = direction / direction_norm
            current_heading = self.velocity / (np.linalg.norm(self.velocity) + 1e-8)
            angle_between = np.arccos(np.clip(np.dot(current_heading, desired_heading), -1.0, 1.0))
            if angle_between < self.max_heading_change * dt:
                new_heading = desired_heading
            else:
                axis = np.cross(current_heading, desired_heading)
                if np.linalg.norm(axis) < 1e-8:
                    new_heading = current_heading
                else:
                    axis = axis / np.linalg.norm(axis)
                    rot_matrix = rotation_matrix(axis, self.max_heading_change * dt)
                    new_heading = np.dot(rot_matrix, current_heading)
            self.velocity = new_heading * np.linalg.norm(self.velocity)
            self.position += self.velocity * dt
            self.last_arrived_point = self.position.copy()
        else:
            # No movement if no waypoint (for base drone)
            pass

def rotation_matrix(axis: np.ndarray, theta: float) -> np.ndarray:
    """Rodrigues' rotation formula for 3D rotation."""
    axis = axis / np.linalg.norm(axis)
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    return np.array([
        [a*a + b*b - c*c - d*d, 2*(b*c - a*d), 2*(b*d + a*c)],
        [2*(b*c + a*d), a*a + c*c - b*b - d*d, 2*(c*d - a*b)],
        [2*(b*d - a*c), 2*(c*d + a*b), a*a + d*d - b*b - c*c]
    ])

class DroneWithPayload(Drone):
    def __init__(self, position: np.ndarray, velocity: np.ndarray, max_accel: float, max_heading_change: float, payloads: list, max_circle_radius: float):
        super().__init__(position, velocity, max_accel, max_heading_change)
        self.payloads = payloads
        self.max_circle_radius = max_circle_radius
        self.circle_radius = 10.0  # initial radius
        self.circle_angle = 0.0

    def step(self, dt: float, waypoint: Optional[np.ndarray]):
        if waypoint is not None:
            super().step(dt, waypoint)
            self.circle_radius = 10.0
            self.circle_angle = 0.0
        else:
            # Move in a circle around last_arrived_point
            self.circle_angle += self.max_heading_change * dt
            self.circle_radius = min(self.circle_radius + 0.5 * dt, self.max_circle_radius)
            # Circle in horizontal plane (NED: x, y)
            x = self.last_arrived_point[0] + self.circle_radius * np.cos(self.circle_angle)
            y = self.last_arrived_point[1] + self.circle_radius * np.sin(self.circle_angle)
            z = self.last_arrived_point[2]
            self.position = np.array([x, y, z])
