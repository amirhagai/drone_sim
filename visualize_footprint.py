import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
# Import 3D plotting tools
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from utils.transformations import get_rotation_matrix
from carried_objects.sensor import Sensor

def visualize_sensor_footprint():
    """
    Calculates and plots the footprint of a sensor on the ground plane.
    Now uses the methods within the Sensor class to perform calculations.
    """
    # 1. --- SENSOR AND ENVIRONMENT DEFINITION ---

    # Define the drone's state in the world (NED coordinates)
    drone_position_ned = np.array([0., 0., -3000.])
    # Let's give the drone a 45-degree heading to the right (positive yaw)
    drone_attitude_rad = np.array([0., 0., np.deg2rad(45)])     # [roll, pitch, yaw]
    
    # Define the sensor's installation angle (pitched 30 degrees down)
    sensor_installation_rad = np.array([0., -np.deg2rad(30), 0.])

    # Create a sensor, passing its installation angles during initialization
    sensor = Sensor(
        installation_angles=sensor_installation_rad,
        resolution=(1920 * 2, 1536 * 9),
        horizontal_fov_deg=8,
        vertical_fov_deg=27
    )

    # Define the Z-plane for the intersection
    z_plane = -100.0

    # 2. --- FOOTPRINT CALCULATION (delegated to sensor object) ---
    print(f"Calculating footprint on plane z={z_plane}...")
    ground_intersections = sensor.calculate_footprint(
        drone_position_ned, 
        drone_attitude_rad, 
        z_plane
    )
    for name, point in ground_intersections.items():
        print(f"  - {name}: Intersects at {np.round(point, 2)} m")


    # 3. --- 3D PLOTTING ---

    if len(ground_intersections) < 2:
        print("\nPlotting skipped: Not enough ground intersection points to form a shape.")
        return

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # --- Plot Drone and Attitude (Body Frame Axes) ---
    # Plot the drone's position
    ax.scatter(drone_position_ned[1], drone_position_ned[0], -drone_position_ned[2], c='black', marker='o', s=100, label='Drone Position')
    
    # Get the rotation matrix to transform body vectors to the world (NED) frame
    C_body_to_ned = get_rotation_matrix(drone_attitude_rad[0], drone_attitude_rad[1], drone_attitude_rad[2])
    
    # Define the body axes vectors (X=Forward, Y=Right, Z=Down)
    axis_length = 500 # Make the vectors 500m long for visibility
    body_x = np.array([axis_length, 0, 0])
    body_y = np.array([0, axis_length, 0])
    body_z = np.array([0, 0, axis_length])
    
    # Transform body axes to world frame
    world_x = C_body_to_ned @ body_x
    world_y = C_body_to_ned @ body_y
    world_z = C_body_to_ned @ body_z

    # Plot the 3 body axes using quiver
    # X - Forward (Red)
    ax.quiver(drone_position_ned[1], drone_position_ned[0], -drone_position_ned[2],
              world_x[1], world_x[0], -world_x[2],
              color='red', label='Body X (Forward)')
    # Y - Right (Green)
    ax.quiver(drone_position_ned[1], drone_position_ned[0], -drone_position_ned[2],
              world_y[1], world_y[0], -world_y[2],
              color='green', label='Body Y (Right)')
    # Z - Down (Blue)
    ax.quiver(drone_position_ned[1], drone_position_ned[0], -drone_position_ned[2],
              world_z[1], world_z[0], -world_z[2],
              color='blue', label='Body Z (Down)')


    # --- Plot Sensor Rays ---
    for name, point in ground_intersections.items():
        # Line from drone to intersection point
        line_x = [drone_position_ned[1], point[1]]
        line_y = [drone_position_ned[0], point[0]]
        line_z = [-drone_position_ned[2], -point[2]]
        ax.plot(line_x, line_y, line_z, 'b--', alpha=0.5)

    # --- Plot Footprint Polygon on the Z-Plane ---
    plot_order = ["bottom_left", "bottom_right", "top_right", "top_left"]
    polygon_vertices_3d = []
    for key in plot_order:
        if key in ground_intersections:
            polygon_vertices_3d.append(ground_intersections[key])
    
    if len(polygon_vertices_3d) >= 3:
        # Note: For plotting, we use Y,X,-Z to match a more intuitive "East, North, Up" view
        plot_verts = [[(v[1], v[0], -v[2]) for v in polygon_vertices_3d]]
        poly = Poly3DCollection(plot_verts, alpha=0.3, facecolors='cyan', edgecolors='k')
        ax.add_collection3d(poly)

    # --- Plot Test Points ---
    if len(polygon_vertices_3d) > 0:
        center_point_3d = np.mean(np.array(polygon_vertices_3d), axis=0)
        is_inside = sensor.is_in_footprint(center_point_3d[:2])
        print(f"\nIs test point {np.round(center_point_3d[:2], 2)} inside footprint? {is_inside}")
        ax.scatter(center_point_3d[1], center_point_3d[0], -center_point_3d[2], c='green', marker='*', s=100, label=f'Test Point (Inside: {is_inside})')

    # --- Configure Axes ---
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_zlabel("Altitude (m)")
    ax.set_title("3D Sensor Footprint Visualization")

    # Set aspect ratio to be equal
    all_points = np.array(list(ground_intersections.values()) + [drone_position_ned])
    x = all_points[:, 1]
    y = all_points[:, 0]
    z = -all_points[:, 2]
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.legend()
    plt.show()

if __name__ == "__main__":
    visualize_sensor_footprint() 