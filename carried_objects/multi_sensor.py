import numpy as np

class MultiSensor:
    def __init__(self, installation_angles_deg):
        self.installation_angles_rad = np.deg2rad(installation_angles_deg)
        self.sensors = []

    def add_sensor(self, sensor):
        self.sensors.append(sensor)

    def step(self, targets_ned_dict, drone_pos_ned, drone_att_rad, dt):
        """
        Updates all child sensors to point at their respective targets.

        Args:
            targets_ned_dict (dict): A dictionary where keys are sensor indices (int)
                                     and values are the target NED coordinates (np.array).
        """
        for i, sensor in enumerate(self.sensors):
            # Check if a specific target exists for this sensor index
            if i in targets_ned_dict:
                target_ned = targets_ned_dict[i]
                sensor.point_at(
                    target_ned,
                    drone_pos_ned,
                    drone_att_rad,
                    dt=dt,
                    multi_sensor_install_rad=self.installation_angles_rad
                ) 