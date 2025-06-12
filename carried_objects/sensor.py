from .base_object import BaseObject
import numpy as np
from utils.transformations import get_rotation_matrix
from matplotlib.path import Path

class Sensor(BaseObject):
    def __init__(self, installation_angles, resolution, horizontal_fov_deg, vertical_fov_deg):
        super().__init__(installation_angles)
        self.resolution_w, self.resolution_h = resolution
        self.hfov_rad = np.deg2rad(horizontal_fov_deg)
        self.vfov_rad = np.deg2rad(vertical_fov_deg)
        self.footprint_path = None
        self.footprint_points = {}
        # Future sensor-specific attributes can go here 

    def get_corner_ray_vectors(self):
        """
        Calculates the direction vectors for the four corners of the sensor's FOV.

        The vectors are in the sensor's own coordinate frame:
        +X: out of the lens, +Y: right, +Z: down.

        Returns:
            A dictionary containing the normalized direction vectors for each corner.
        """
        # Tangent of half the FOV gives the extent of the image plane at 1 unit distance
        tan_half_hfov = np.tan(self.hfov_rad / 2.0)
        tan_half_vfov = np.tan(self.vfov_rad / 2.0)

        # The corners of the image plane at 1 unit distance
        # The forward component (x) is 1. The y and z components are based on the tan of the angles.
        top_left_vec = np.array([1.0, -tan_half_hfov, -tan_half_vfov])
        top_right_vec = np.array([1.0, tan_half_hfov, -tan_half_vfov])
        bottom_left_vec = np.array([1.0, -tan_half_hfov, tan_half_vfov])
        bottom_right_vec = np.array([1.0, tan_half_hfov, tan_half_vfov])

        # Normalize the vectors to get unit direction vectors (rays)
        return {
            "top_left": top_left_vec / np.linalg.norm(top_left_vec),
            "top_right": top_right_vec / np.linalg.norm(top_right_vec),
            "bottom_left": bottom_left_vec / np.linalg.norm(bottom_left_vec),
            "bottom_right": bottom_right_vec / np.linalg.norm(bottom_right_vec),
        } 

    def calculate_footprint(self, drone_position_ned, drone_attitude_rad, z_plane=0.0, platform_install_rad=None):
        """
        Calculates the sensor's footprint on a given Z-plane.

        Args:
            drone_position_ned (np.array): The drone's position [N, E, D].
            drone_attitude_rad (np.array): The drone's attitude [roll, pitch, yaw].
            z_plane (float): The Z-value of the intersection plane in the NED frame.
            platform_install_rad (np.array, optional): The installation angles of the platform
                                                      the sensor is mounted on, relative to the drone body.
                                                      Defaults to None.

        Returns:
            A dictionary of the 3D intersection points for the corners that hit the plane.
        """
        # The full rotation is Sensor -> Platform -> Body -> NED.
        # self.installation_angles are Sensor -> Platform.
        # platform_install_rad are Platform -> Body.
        # drone_attitude_rad are Body -> NED.
        C_sensor_to_platform = get_rotation_matrix(*self.installation_angles)
        
        C_platform_to_body = np.identity(3)
        if platform_install_rad is not None:
            C_platform_to_body = get_rotation_matrix(*platform_install_rad)

        # Get rotation from body frame -> NED frame
        C_body_to_ned = get_rotation_matrix(*drone_attitude_rad)

        corner_rays = self.get_corner_ray_vectors()
        intersections = {}

        for name, ray_sensor in corner_rays.items():
            ray_platform = C_sensor_to_platform @ ray_sensor
            ray_body = C_platform_to_body @ ray_platform
            ray_ned = C_body_to_ned @ ray_body
            
            # Ensure the ray points towards the plane (V.z must have opposite sign to plane position)
            # For z_plane=0, we need V.z > 0. For z_plane=-200, we need V.z > 0.
            # In general, if P0.z is above plane, V.z must be positive.
            if ray_ned[2] > 1e-9: # Check if ray points down at all
                # Ray equation P(t) = P0 + t*V. We solve for t where P(t).z = z_plane.
                # P0[2] + t*V[2] = z_plane  =>  t = (z_plane - P0[2]) / V[2]
                t = (z_plane - drone_position_ned[2]) / ray_ned[2]
                if t > 0: # Ensure intersection is in front of the sensor
                    intersection_point = drone_position_ned + t * ray_ned
                    intersections[name] = intersection_point

        self.footprint_points = intersections
        
        # Create a Path object for the point-in-polygon test, if possible
        plot_order = ["bottom_left", "bottom_right", "top_right", "top_left"]
        polygon_points = []
        for key in plot_order:
            if key in self.footprint_points:
                # We only need the (x,y) or (N,E) coordinates for a 2D check
                polygon_points.append(self.footprint_points[key][:2])
        
        if len(polygon_points) >= 3:
            self.footprint_path = Path(polygon_points)
        else:
            self.footprint_path = None # Not enough points to form a polygon

        return self.footprint_points

    def is_in_footprint(self, point_xy):
        """
        Checks if a 2D point (X, Y) lies within the last calculated footprint.

        Args:
            point_xy (tuple or list): The (x, y) coordinates of the point to check.

        Returns:
            True if the point is inside the polygon, False otherwise.
        """
        if self.footprint_path is None:
            # Cannot check if a valid polygon was not formed
            return False
        
        return self.footprint_path.contains_point(point_xy) 