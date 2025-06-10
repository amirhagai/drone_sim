from abc import ABC, abstractmethod
import numpy as np

class BaseDrone(ABC):
    def __init__(self, position, velocity, acceleration, max_heading_change_rate, horsepower, max_static_thrust):
        self.position = np.array(position, dtype=float)  # NED coordinates
        self.velocity = np.array(velocity, dtype=float)  # NED coordinates
        self.acceleration = np.array(acceleration, dtype=float) # NED coordinates
        self.attitude = np.zeros(3)  # [roll, pitch, yaw]
        self.max_heading_change_rate = max_heading_change_rate # rad/s
        self.horsepower = horsepower
        self.max_static_thrust = max_static_thrust # Newtons
        self.carried_objects = []

    @abstractmethod
    def step(self, dt, waypoint=None):
        pass

    def _calculate_thrust(self):
        """Calculates current thrust based on horsepower and velocity."""
        speed = np.linalg.norm(self.velocity)
        if speed > 0.1: # Avoid division by zero and handle low speed case
            power_in_watts = self.horsepower * 745.7
            dynamic_thrust = power_in_watts / speed
            # Return the lesser of calculated thrust or the max physical static thrust
            return min(dynamic_thrust, self.max_static_thrust)
        else:
            # At low/zero speed, use max static thrust
            return self.max_static_thrust

    def update_physics(self, dt):
        """
        Updates the drone's position and velocity based on its acceleration.
        A simple Euler integration is used here.
        """
        self.velocity += self.acceleration * dt
        self.position += self.velocity * dt 