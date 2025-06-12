import numpy as np
from .base_drone import BaseDrone

class AdvancedDrone(BaseDrone):
    def __init__(self, position, velocity, acceleration, max_heading_change_rate,
                 horsepower, max_static_thrust, max_circling_radius=100.0, circling_radius_rate=1.0):
        super().__init__(position, velocity, acceleration, max_heading_change_rate, horsepower, max_static_thrust)
        
        self.circling = False
        self.circling_center = np.copy(position)
        self.circling_radius = 0.0
        self.max_circling_radius = max_circling_radius
        self.circling_radius_rate = circling_radius_rate
        self.circling_angle = 0.0
        self.last_waypoint_reached = True

    def add_object(self, obj):
        self.carried_objects.append(obj)

    def step(self, dt, waypoint=None):
        if waypoint is not None:
            self.circling = False
            self.last_waypoint_reached = False
            waypoint = np.array(waypoint, dtype=float)
            
            if np.linalg.norm(waypoint - self.position) < 0.5: # Waypoint reached
                self.last_waypoint_reached = True
                self.circling_center = np.copy(self.position)
                self.acceleration = np.zeros(3) # Stop accelerating
                self.velocity = np.zeros(3)     # Stop moving
            else:
                self.fly_towards(dt, waypoint)

        else: # No waypoint provided
            if not self.circling:
                self.circling = True
                self.circling_center = np.copy(self.position)
                self.circling_radius = 0.0
                self.circling_angle = self.attitude[2] # Start circling from current heading

            if self.circling:
                temp_waypoint = self.get_circling_point(dt)
                self.fly_towards(dt, temp_waypoint)

        self.update_physics(dt)

    def get_circling_point(self, dt):
        if self.circling_radius < self.max_circling_radius:
            self.circling_radius += self.circling_radius_rate * dt
        
        speed = np.linalg.norm(self.velocity)
        if speed < 2: speed = 5 # default speed if slow/stopped
        
        angular_rate = 0
        if self.circling_radius > 0.1:
            angular_rate = speed / self.circling_radius
        else:
            angular_rate = 1 # rad/s, to start the circle
        
        self.circling_angle += angular_rate * dt

        lookahead_seconds = 2.0
        lookahead_angle = self.circling_angle + angular_rate * lookahead_seconds
        
        temp_waypoint_x = self.circling_center[0] + self.circling_radius * np.cos(lookahead_angle)
        temp_waypoint_y = self.circling_center[1] + self.circling_radius * np.sin(lookahead_angle)
        temp_waypoint_z = self.circling_center[2]
        return np.array([temp_waypoint_x, temp_waypoint_y, temp_waypoint_z])

    def fly_towards(self, dt, waypoint):
        direction_to_waypoint = waypoint - self.position
        
        if np.linalg.norm(direction_to_waypoint) < 0.1:
            self.acceleration = np.zeros(3)
            return

        desired_yaw = np.arctan2(direction_to_waypoint[1], direction_to_waypoint[0])
        horizontal_dist = np.linalg.norm(direction_to_waypoint[:2])
        desired_pitch = -np.arctan2(direction_to_waypoint[2], horizontal_dist)

        current_yaw = self.attitude[2]
        yaw_error = desired_yaw - current_yaw
        while yaw_error > np.pi: yaw_error -= 2 * np.pi
        while yaw_error < -np.pi: yaw_error += 2 * np.pi
        max_yaw_change = self.max_heading_change_rate * dt
        yaw_change = np.clip(yaw_error, -max_yaw_change, max_yaw_change)
        self.attitude[2] += yaw_change

        current_pitch = self.attitude[1]
        pitch_error = desired_pitch - current_pitch
        while pitch_error > np.pi: pitch_error -= 2 * np.pi
        while pitch_error < -np.pi: pitch_error += 2 * np.pi
        max_pitch_change = self.max_heading_change_rate * dt
        pitch_change = np.clip(pitch_error, -max_pitch_change, max_pitch_change)
        self.attitude[1] += pitch_change
        
        self.attitude[0] = 0

        thrust = self._calculate_thrust()
        pitch, yaw = self.attitude[1], self.attitude[2]
        self.acceleration[0] = thrust * np.cos(yaw) * np.cos(pitch)
        self.acceleration[1] = thrust * np.sin(yaw) * np.cos(pitch)
        self.acceleration[2] = -thrust * np.sin(pitch) 