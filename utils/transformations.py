import numpy as np

def get_rotation_matrix(roll, pitch, yaw):
    """
    Calculates the rotation matrix to transform a vector from a body frame to a parent (NED) frame.
    The rotation sequence is ZYX (yaw, pitch, roll).
    """
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)

    # Rotation matrices for each axis
    R_x = np.array([[1, 0, 0],
                    [0, cr, -sr],
                    [0, sr, cr]])

    R_y = np.array([[cp, 0, sp],
                    [0, 1, 0],
                    [-sp, 0, cp]])

    R_z = np.array([[cy, -sy, 0],
                    [sy, cy, 0],
                    [0, 0, 1]])
    
    # Combined rotation matrix for ZYX extrinsic rotation.
    # This transforms from the body frame to the parent frame.
    R = R_z @ R_y @ R_x
    return R

def ned_to_body_frame(vector_ned, roll, pitch, yaw):
    """
    Converts a vector from NED to the body frame.
    """
    # Combined rotation matrix. Z-Y-X rotation sequence (yaw, pitch, roll)
    # The rotation from NED to body is the transpose of body to NED.
    R = get_rotation_matrix(roll, pitch, yaw).T
    return R @ vector_ned

def body_to_ned_frame(vector_body, roll, pitch, yaw):
    """
    Converts a vector from the body frame to NED.
    """
    R = get_rotation_matrix(roll, pitch, yaw)
    return R @ vector_body

def normalize_vector(vector):
    """
    Normalizes a vector.
    """
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm 