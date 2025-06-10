from .base_object import BaseObject

class Sensor(BaseObject):
    def __init__(self, installation_angles):
        super().__init__(installation_angles)
        # Future sensor-specific attributes can go here 