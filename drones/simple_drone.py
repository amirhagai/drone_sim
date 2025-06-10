import numpy as np
from .base_drone import BaseDrone

class SimpleDrone(BaseDrone):

    def __init__(self, position, velocity, acceleration, max_heading_change_rate, horsepower, max_static_thrust):
        super().__init__(position, velocity, acceleration, max_heading_change_rate, horsepower, max_static_thrust)

    def step(self, dt, waypoint=None):
        if waypoint is not None:
            waypoint = np.array(waypoint, dtype=float)
            direction_to_waypoint = waypoint - self.position
            distance_to_waypoint = np.linalg.norm(direction_to_waypoint)

            if distance_to_waypoint > 0.1:
                # Calculate desired yaw and pitch to point at the waypoint
                desired_yaw = np.arctan2(direction_to_waypoint[1], direction_to_waypoint[0])
                horizontal_dist = np.linalg.norm(direction_to_waypoint[:2])
                desired_pitch = -np.arctan2(direction_to_waypoint[2], horizontal_dist)

                # --- Update Yaw (Heading) ---
                current_yaw = self.attitude[2]
                yaw_error = desired_yaw - current_yaw
                # Normalize angle
                while yaw_error > np.pi: yaw_error -= 2 * np.pi
                while yaw_error < -np.pi: yaw_error += 2 * np.pi

                max_yaw_change = self.max_heading_change_rate * dt
                yaw_change = np.clip(yaw_error, -max_yaw_change, max_yaw_change)
                self.attitude[2] += yaw_change

                # --- Update Pitch ---
                # Using same change rate limit for pitch
                current_pitch = self.attitude[1]
                pitch_error = desired_pitch - current_pitch
                # Normalize angle
                while pitch_error > np.pi: pitch_error -= 2 * np.pi
                while pitch_error < -np.pi: pitch_error += 2 * np.pi

                max_pitch_change = self.max_heading_change_rate * dt # Assuming same rate
                pitch_change = np.clip(pitch_error, -max_pitch_change, max_pitch_change)
                self.attitude[1] += pitch_change
                
                # We assume no roll for this simple drone
                self.attitude[0] = 0

                # Update acceleration vector to be aligned with drone's new orientation
                thrust = self._calculate_thrust()
                pitch, yaw = self.attitude[1], self.attitude[2]
                self.acceleration[0] = thrust * np.cos(yaw) * np.cos(pitch)
                self.acceleration[1] = thrust * np.sin(yaw) * np.cos(pitch)
                self.acceleration[2] = -thrust * np.sin(pitch)

        # If no waypoint, it will just continue with its last acceleration
        self.update_physics(dt) 