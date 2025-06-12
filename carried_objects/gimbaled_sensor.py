import numpy as np
from .sensor import Sensor
from utils.transformations import get_rotation_matrix, normalize_vector

class GimbaledSensor(Sensor):
    def __init__(self, installation_angles, resolution, horizontal_fov_deg, vertical_fov_deg,
                 max_gimbal_rate_dps, max_gimbal_yaw_deg, max_gimbal_pitch_deg):
        
        # This is the sensor's own orientation *within the MultiSensor platform*.
        # We pass it to the parent Sensor's __init__ to handle the FOV ray calculations.
        super().__init__(installation_angles, resolution, horizontal_fov_deg, vertical_fov_deg)

        # Gimbal properties
        self.gimbal_angles_rad = np.array([0.0, 0.0])  # [yaw, pitch]
        self.max_gimbal_rate_rps = np.deg2rad(max_gimbal_rate_dps)
        self.max_gimbal_angles_rad = np.array([
            np.deg2rad(max_gimbal_yaw_deg),
            np.deg2rad(max_gimbal_pitch_deg)
        ])

    def point_at(self, target_ned, drone_pos_ned, drone_att_rad, dt, multi_sensor_install_rad=None):
        """
        Updates the gimbal angles to track a target point.
        """
        # 1. Find the target direction vector in the world (NED) frame.
        direction_to_target_ned = normalize_vector(target_ned - drone_pos_ned)

        # 2. Transform this world vector into the MultiSensor's local coordinate frame.
        # This requires rotating it by the inverse (transpose) of the drone and platform rotations.
        C_body_to_ned = get_rotation_matrix(*drone_att_rad)
        C_platform_to_body = get_rotation_matrix(*multi_sensor_install_rad)
        
        direction_in_body = C_body_to_ned.T @ direction_to_target_ned
        direction_in_platform = C_platform_to_body.T @ direction_in_body

        # 3. Calculate the desired gimbal angles to point at this local direction vector.
        desired_yaw = np.arctan2(direction_in_platform[1], direction_in_platform[0])
        desired_pitch = -np.arctan2(direction_in_platform[2], np.linalg.norm(direction_in_platform[:2]))
        desired_angles = np.array([desired_yaw, desired_pitch])

        # 4. Calculate the error and apply the rate limit.
        error = desired_angles - self.gimbal_angles_rad
        # Normalize angle errors to the shortest path
        for i in range(2):
            while error[i] > np.pi: error[i] -= 2 * np.pi
            while error[i] < -np.pi: error[i] += 2 * np.pi
        
        max_change = self.max_gimbal_rate_rps * dt
        change = np.clip(error, -max_change, max_change)
        
        # --- DIAGNOSTIC PRINT for Rate Limit ---
        # if np.any(np.abs(error) > np.abs(change) + 1e-6): # Check if clipping occurred
            # print(f"**GIMBAL RATE LIMITED**: "
            #       f"Desired: {np.round(np.rad2deg(error), 2)} deg, "
            #       f"Actual: {np.round(np.rad2deg(change), 2)} deg")

        # 5. Apply the change and then enforce the hard travel limits.
        new_angles = self.gimbal_angles_rad + change
        self.gimbal_angles_rad = np.clip(
            new_angles,
            -self.max_gimbal_angles_rad,
            self.max_gimbal_angles_rad
        )

        # --- DIAGNOSTIC PRINT for Travel Limit ---
        if np.any(np.abs(self.gimbal_angles_rad) >= self.max_gimbal_angles_rad - 1e-6):
             # Check if we are at the edge of the travel limit
            at_limit_yaw = np.abs(self.gimbal_angles_rad[0]) >= self.max_gimbal_angles_rad[0] - 1e-6
            at_limit_pitch = np.abs(self.gimbal_angles_rad[1]) >= self.max_gimbal_angles_rad[1] - 1e-6
            # if at_limit_yaw or at_limit_pitch:
            #      print(f"**GIMBAL TRAVEL LIMITED**: "
            #            f"Yaw: {np.round(np.rad2deg(self.gimbal_angles_rad[0]), 1)}° "
            #            f"(Max: {np.round(np.rad2deg(self.max_gimbal_angles_rad[0]), 1)}°), "
            #            f"Pitch: {np.round(np.rad2deg(self.gimbal_angles_rad[1]), 1)}° "
            #            f"(Max: {np.round(np.rad2deg(self.max_gimbal_angles_rad[1]), 1)}°)")

    def calculate_footprint(self, drone_position_ned, drone_attitude_rad, z_plane=0.0, platform_install_rad=None):
        """
        Override of the original calculate_footprint to include the gimbal rotation.
        """
        # Get the rotation matrices for each step of the transformation
        C_gimbal_to_platform = get_rotation_matrix(0, self.gimbal_angles_rad[1], self.gimbal_angles_rad[0])
        C_platform_to_body = get_rotation_matrix(*platform_install_rad)
        C_body_to_ned = get_rotation_matrix(*drone_attitude_rad)

        # Get the initial rays in the sensor's own frame (assuming it points forward from the gimbal)
        corner_rays_sensor_frame = self.get_corner_ray_vectors()
        intersections = {}

        for name, ray_sensor in corner_rays_sensor_frame.items():
            # Chain the rotations: sensor -> gimbal -> platform -> body -> world
            ray_platform = C_gimbal_to_platform @ ray_sensor
            ray_body = C_platform_to_body @ ray_platform
            ray_ned = C_body_to_ned @ ray_body

            # Perform ray-plane intersection as before
            if ray_ned[2] > 1e-9:
                t = (z_plane - drone_position_ned[2]) / ray_ned[2]
                if t > 0:
                    intersection_point = drone_position_ned + t * ray_ned
                    intersections[name] = intersection_point
        
        self.footprint_points = intersections
        return self.footprint_points 