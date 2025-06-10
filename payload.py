import numpy as np

class Payload:
    def __init__(self, name: str, installation_angles: tuple):
        self.name = name
        self.installation_angles = installation_angles  # (yaw, pitch, roll) in radians

class Sensor(Payload):
    def __init__(self, installation_angles: tuple):
        super().__init__('Sensor', installation_angles)

class RocketLauncher(Payload):
    def __init__(self, installation_angles: tuple):
        super().__init__('RocketLauncher', installation_angles)
