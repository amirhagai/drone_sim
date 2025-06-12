import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import imageio
import os
import shutil

from drones.advanced_drone import AdvancedDrone
from carried_objects.sensor import Sensor
from utils.transformations import get_rotation_matrix

def visualize_sensor_footprint_on_advanced_drone():
    """
    Generates a GIF of the AdvancedDrone flying and its sensor footprint changing over time.
    """
    # --- Animation & Simulation Settings ---
    gif_filename = "advanced_drone_animation.gif"
    temp_frame_dir = "temp_frames_adv"
    simulation_duration = 30  # seconds
    dt = 0.5  # time step for simulation and frame rate

    # Create a temporary directory for frames
    if os.path.exists(temp_frame_dir):
        shutil.rmtree(temp_frame_dir)
    os.makedirs(temp_frame_dir)
    
    print(f"Generating frames in '{temp_frame_dir}'...")
    filenames = []

    # --- Drone and Sensor Initialization ---
    advanced_drone = AdvancedDrone(
        position=[0, 0, -3000],
        velocity=[50, 0, 0],
        acceleration=[0, 0, 0],
        max_heading_change_rate=np.deg2rad(90),
        horsepower=10.0,
        max_static_thrust=500.0,
        max_circling_radius=1000,
        circling_radius_rate=10.0
    )
    sensor = Sensor(
        installation_angles=np.array([0., -np.deg2rad(30), 0.]), # Pitched 30 deg down
        resolution=(1920, 1080),
        horizontal_fov_deg=8,
        vertical_fov_deg=27
    )
    z_plane = -100.0
    waypoint = np.array([2000, 2000, -2800])

    # --- Pre-computation for Axis Limits ---
    print("Pre-computing flight path to determine axis bounds...")
    all_points_for_bounds = []
    # Create a temporary drone instance for the pre-run
    temp_drone = AdvancedDrone(
        position=np.copy(advanced_drone.position),
        velocity=np.copy(advanced_drone.velocity),
        acceleration=np.copy(advanced_drone.acceleration),
        max_heading_change_rate=advanced_drone.max_heading_change_rate,
        horsepower=advanced_drone.horsepower,
        max_static_thrust=advanced_drone.max_static_thrust,
        max_circling_radius=advanced_drone.max_circling_radius,
        circling_radius_rate=advanced_drone.circling_radius_rate
    )

    time_steps = np.arange(0, simulation_duration, dt)
    for t in time_steps:
        current_waypoint_pre = waypoint if t < simulation_duration / 2 else None
        temp_drone.step(dt, current_waypoint_pre)
        all_points_for_bounds.append(temp_drone.position)
        intersections = sensor.calculate_footprint(temp_drone.position, temp_drone.attitude, z_plane)
        if intersections:
            all_points_for_bounds.extend(list(intersections.values()))
            
    all_points_for_bounds = np.array(all_points_for_bounds)
    min_coords = all_points_for_bounds.min(axis=0)
    max_coords = all_points_for_bounds.max(axis=0)
    
    # Add a 10% padding to the bounds
    x_range = max_coords[1] - min_coords[1]
    y_range = max_coords[0] - min_coords[0]
    xlim = [min_coords[1] - 0.1 * x_range, max_coords[1] + 0.1 * x_range]
    ylim = [min_coords[0] - 0.1 * y_range, max_coords[0] + 0.1 * y_range]
    zlim = [0, -min_coords[2] + 200] # Z-limit based on max altitude

    # --- Simulation Loop ---
    for i, t in enumerate(time_steps):
        # Update drone state
        # For the second half of the simulation, remove the waypoint to trigger circling
        current_waypoint = waypoint if t < simulation_duration / 2 else None
        advanced_drone.step(dt, current_waypoint)

        # Get drone state for this frame
        drone_pos = advanced_drone.position
        drone_att = advanced_drone.attitude

        # Calculate footprint
        ground_intersections = sensor.calculate_footprint(drone_pos, drone_att, z_plane)

        # --- Plotting ---
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(drone_pos[1], drone_pos[0], -drone_pos[2], c='black', marker='o', s=100, label='Drone')
        C_body_to_ned = get_rotation_matrix(drone_att[0], drone_att[1], drone_att[2])
        axis_length = 500
        body_x, body_y, body_z = np.array([axis_length, 0, 0]), np.array([0, axis_length, 0]), np.array([0, 0, axis_length])
        world_x, world_y, world_z = C_body_to_ned @ body_x, C_body_to_ned @ body_y, C_body_to_ned @ body_z
        ax.quiver(drone_pos[1], drone_pos[0], -drone_pos[2], world_x[1], world_x[0], -world_x[2], color='red')
        ax.quiver(drone_pos[1], drone_pos[0], -drone_pos[2], world_y[1], world_y[0], -world_y[2], color='green')
        ax.quiver(drone_pos[1], drone_pos[0], -drone_pos[2], world_z[1], world_z[0], -world_z[2], color='blue')

        for name, point in ground_intersections.items():
            ax.plot([drone_pos[1], point[1]], [drone_pos[0], point[0]], [-drone_pos[2], -point[2]], 'c--', alpha=0.4)

        plot_order = ["bottom_left", "bottom_right", "top_right", "top_left"]
        polygon_vertices_3d = [ground_intersections[key] for key in plot_order if key in ground_intersections]
        if len(polygon_vertices_3d) >= 3:
            plot_verts = [[(v[1], v[0], -v[2]) for v in polygon_vertices_3d]]
            ax.add_collection3d(Poly3DCollection(plot_verts, alpha=0.3, facecolors='cyan', edgecolors='k'))
        
        ax.set_xlabel("East (m)"); ax.set_ylabel("North (m)"); ax.set_zlabel("Altitude (m)")
        ax.set_title(f"Advanced Drone Flight (Time: {t:.1f}s)")
        
        # Apply the pre-computed static limits
        ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_zlim(zlim)

        ax.view_init(elev=50, azim=-45) # Set a consistent camera angle

        # Save frame
        filename = f"{temp_frame_dir}/frame_{i:03d}.png"
        filenames.append(filename)
        plt.savefig(filename)
        plt.close(fig)
        print(f"  - Saved frame {i+1}/{len(time_steps)} for T={t:.1f}s")

    # --- Compile GIF ---
    print(f"\nCompiling frames into '{gif_filename}'...")
    with imageio.get_writer(gif_filename, mode='I', duration=dt, loop=0) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
    
    print("Cleaning up temporary frame files...")
    shutil.rmtree(temp_frame_dir)
    print(f"\nSuccessfully created {gif_filename}")

if __name__ == "__main__":
    visualize_sensor_footprint_on_advanced_drone() 