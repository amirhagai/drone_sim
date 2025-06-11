import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import imageio
import os
import shutil

from carried_objects.sensor import Sensor
from utils.transformations import get_rotation_matrix

def create_animation():
    """
    Generates a series of 3D plots for different yaw angles and compiles them into a GIF.
    """
    # --- Animation Settings ---
    yaw_angles_deg = np.arange(0, 50, 5) # 0 to 45 degrees in 5-degree steps
    gif_filename = "footprint_animation.gif"
    temp_frame_dir = "temp_frames"
    
    # Create a temporary directory for frames if it doesn't exist
    if os.path.exists(temp_frame_dir):
        shutil.rmtree(temp_frame_dir)
    os.makedirs(temp_frame_dir)
    
    print(f"Generating frames in '{temp_frame_dir}'...")
    filenames = []

    # --- Sensor and Environment Definition (remains constant) ---
    drone_position_ned = np.array([0., 0., -3000.])
    sensor_installation_rad = np.array([0., -np.deg2rad(30), 0.])
    z_plane = -100.0
    sensor = Sensor(
        installation_angles=sensor_installation_rad,
        resolution=(1920 * 2, 1536 * 9),
        horizontal_fov_deg=8,
        vertical_fov_deg=27
    )
    
    # --- Loop Through Angles and Generate Frames ---
    for i, yaw_deg in enumerate(yaw_angles_deg):
        drone_attitude_rad = np.array([0., 0., np.deg2rad(yaw_deg)])

        ground_intersections = sensor.calculate_footprint(
            drone_position_ned, drone_attitude_rad, z_plane
        )

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plotting logic is the same as in visualize_footprint.py
        ax.scatter(drone_position_ned[1], drone_position_ned[0], -drone_position_ned[2], c='black', marker='o', s=100, label='Drone Position')
        C_body_to_ned = get_rotation_matrix(drone_attitude_rad[0], drone_attitude_rad[1], drone_attitude_rad[2])
        axis_length = 500
        body_x, body_y, body_z = np.array([axis_length, 0, 0]), np.array([0, axis_length, 0]), np.array([0, 0, axis_length])
        world_x, world_y, world_z = C_body_to_ned @ body_x, C_body_to_ned @ body_y, C_body_to_ned @ body_z
        ax.quiver(drone_position_ned[1], drone_position_ned[0], -drone_position_ned[2], world_x[1], world_x[0], -world_x[2], color='red', label='Body X (Forward)')
        ax.quiver(drone_position_ned[1], drone_position_ned[0], -drone_position_ned[2], world_y[1], world_y[0], -world_y[2], color='green', label='Body Y (Right)')
        ax.quiver(drone_position_ned[1], drone_position_ned[0], -drone_position_ned[2], world_z[1], world_z[0], -world_z[2], color='blue', label='Body Z (Down)')

        for name, point in ground_intersections.items():
            ax.plot([drone_position_ned[1], point[1]], [drone_position_ned[0], point[0]], [-drone_position_ned[2], -point[2]], 'b--', alpha=0.5)

        plot_order = ["bottom_left", "bottom_right", "top_right", "top_left"]
        polygon_vertices_3d = [ground_intersections[key] for key in plot_order if key in ground_intersections]
        if len(polygon_vertices_3d) >= 3:
            plot_verts = [[(v[1], v[0], -v[2]) for v in polygon_vertices_3d]]
            ax.add_collection3d(Poly3DCollection(plot_verts, alpha=0.3, facecolors='cyan', edgecolors='k'))
        
        ax.set_xlabel("East (m)"); ax.set_ylabel("North (m)"); ax.set_zlabel("Altitude (m)")
        ax.set_title(f"3D Sensor Footprint Visualization (Yaw: {yaw_deg}°)")

        # Set consistent axis limits for all frames
        ax.set_xlim(-4000, 4000); ax.set_ylim(-4000, 4000); ax.set_zlim(0, 3500)

        # Save the frame
        filename = f"{temp_frame_dir}/frame_{i:02d}.png"
        filenames.append(filename)
        plt.savefig(filename)
        plt.close(fig) # Close the figure to free memory
        print(f"  - Saved frame {i+1}/{len(yaw_angles_deg)} for yaw {yaw_deg}°")

    # --- Compile GIF ---
    print(f"\nCompiling frames into '{gif_filename}'...")
    with imageio.get_writer(gif_filename, mode='I', duration=0.5) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
    
    # --- Cleanup ---
    print("Cleaning up temporary frame files...")
    shutil.rmtree(temp_frame_dir)
    
    print(f"\nSuccessfully created {gif_filename}")

if __name__ == "__main__":
    create_animation() 