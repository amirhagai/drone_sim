import numpy as np

def ned_to_body_frame(vector_ned, roll, pitch, yaw):
    """
    Converts a vector from NED to the body frame.
    """
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)

    # Rotation matrix from NED to body frame
    R_x = np.array([[1, 0, 0],
                    [0, cr, sr],
                    [0, -sr, cr]])

    R_y = np.array([[cp, 0, -sp],
                    [0, 1, 0],
                    [sp, 0, cp]])

    R_z = np.array([[cy, sy, 0],
                    [-sy, cy, 0],
                    [0, 0, 1]])
    
    # Combined rotation matrix. Z-Y-X rotation sequence (yaw, pitch, roll)
    R = R_x @ R_y @ R_z
    return R @ vector_ned

def body_to_ned_frame(vector_body, roll, pitch, yaw):
    """
    Converts a vector from the body frame to NED.
    """
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)

    # Rotation matrix from body to NED is the transpose of NED to body
    R_x = np.array([[1, 0, 0],
                    [0, cr, -sr],
                    [0, sr, cr]])

    R_y = np.array([[cp, 0, sp],
                    [0, 1, 0],
                    [-sp, 0, cp]])

    R_z = np.array([[cy, -sy, 0],
                    [sy, cy, 0],
                    [0, 0, 1]])

    # Combined rotation matrix. Z-Y-X rotation sequence (yaw, pitch, roll)
    # The rotation from body to NED is R_z.T @ R_y.T @ R_x.T = (R_x @ R_y @ R_z).T
    R = R_z.T @ R_y.T @ R_x.T
    return R @ vector_body

def normalize_vector(vector):
    """
    Normalizes a vector.
    """
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm 