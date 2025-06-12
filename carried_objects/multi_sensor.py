import numpy as np

class MultiSensor:
    def __init__(self, installation_angles_deg):
        self.installation_angles_rad = np.deg2rad(installation_angles_deg)
        self.sensors = []

    def add_sensor(self, sensor):
        self.sensors.append(sensor)

    def step(self, target_ned, drone_pos_ned, drone_att_rad, dt):
        """
        Updates all child sensors to point at the target.
        """
        for sensor in self.sensors:
            sensor.point_at(
                target_ned,
                drone_pos_ned,
                drone_att_rad,
                self.installation_angles_rad,
                dt
            ) 