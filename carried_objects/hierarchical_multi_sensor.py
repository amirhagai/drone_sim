import numpy as np
from carried_objects.sensor import Sensor
from carried_objects.gimbaled_sensor import GimbaledSensor

class HierarchicalMultiSensor:
    def __init__(self, installation_angles_deg, parent_sensor: Sensor, child_sensor: GimbaledSensor):
        self.installation_angles_rad = np.deg2rad(installation_angles_deg)
        self.parent_sensor = parent_sensor
        self.child_sensor = child_sensor

        # --- NEW: Set child gimbal limits based on FOV difference ---
        # The user is correct, this is a much more elegant way to constrain the child.
        if parent_sensor.hfov_rad > child_sensor.hfov_rad:
            max_yaw = (parent_sensor.hfov_rad - child_sensor.hfov_rad) / 2.0
        else:
            max_yaw = 0.0 # Child is wider than parent, so it can't move at all.
            print("Warning: Child sensor HFOV is wider than parent's. Yaw gimbal travel will be zero.")

        if parent_sensor.vfov_rad > child_sensor.vfov_rad:
            max_pitch = (parent_sensor.vfov_rad - child_sensor.vfov_rad) / 2.0
        else:
            max_pitch = 0.0
            print("Warning: Child sensor VFOV is wider than parent's. Pitch gimbal travel will be zero.")

        # We take the *minimum* of the sensor's own physical limits and the new hierarchical constraint.
        new_max_gimbal_angles = np.array([max_yaw, max_pitch])
        original_limits = np.copy(self.child_sensor.max_gimbal_angles_rad)
        
        self.child_sensor.max_gimbal_angles_rad = np.minimum(
            original_limits,
            new_max_gimbal_angles
        )
        print(f"Hierarchical constraint applied. Original child gimbal limits (deg): "
              f"Yaw={np.rad2deg(original_limits[0]):.1f}, "
              f"Pitch={np.rad2deg(original_limits[1]):.1f}")
        print(f"New child gimbal limits (deg): "
              f"Yaw={np.rad2deg(self.child_sensor.max_gimbal_angles_rad[0]):.1f}, "
              f"Pitch={np.rad2deg(self.child_sensor.max_gimbal_angles_rad[1]):.1f}")


    def step(self, target_ned, drone_pos_ned, drone_att_rad, dt, z_plane=0.0):
        """
        Updates the parent and child sensors in sequence.
        The parent-child footprint constraint is handled by pre-setting
        the child's gimbal limits during initialization.
        """
        # 1. First, update the static parent sensor to get its latest footprint for visualization.
        # self.parent_sensor.calculate_footprint(
        #     drone_pos_ned,
        #     drone_att_rad,
        #     z_plane,
        #     platform_install_rad=self.installation_angles_rad
        # )

        # 2. Then, command the child sensor to point at the target using its standard method.
        #    The gimbal limits have already been constrained.
        self.child_sensor.point_at(
            target_ned,
            drone_pos_ned,
            drone_att_rad,
            dt=dt,
            multi_sensor_install_rad=self.installation_angles_rad
        ) 