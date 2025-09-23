import numpy as np
from scipy.spatial.transform import Rotation as R

def normalize(v):
    """Normalize a vector safely."""
    norm = np.linalg.norm(v)
    if norm < 1e-9:
        raise ValueError(f"Zero-length vector encountered: {v}")
    return v / norm

def construct_new_frame(p_origin, p_x, p_xy):
    """
    Construct matrix for 'new' frame in 'base'.
    - p_origin: origin of new frame (3D point)
    - p_x: point defining X direction
    - p_xy: point defining XY plane
    """

    x_axis = normalize(p_x - p_origin)
    y_prov = p_xy - p_origin
    z_axis = normalize(np.cross(x_axis, y_prov))
    y_axis = np.cross(z_axis, x_axis)

    Rotation = np.column_stack((x_axis, y_axis, z_axis))
    
    Transform = np.eye(4)
    Transform[:3, :3] = Rotation
    Transform[:3, 3] = p_origin
    return Transform
