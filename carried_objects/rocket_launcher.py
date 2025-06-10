from .base_object import BaseObject

class RocketLauncher(BaseObject):
    def __init__(self, installation_angles):
        super().__init__(installation_angles)
        # Future rocket-launcher-specific attributes can go here 