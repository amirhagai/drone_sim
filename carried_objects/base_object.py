import numpy as np

class BaseObject:
    def __init__(self, installation_angles):
        # [yaw, pitch, roll] relative to drone body frame
        self.installation_angles = np.array(installation_angles, dtype=float) 