import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import imageio
import os
import shutil

from drones.advanced_drone import AdvancedDrone
from carried_objects.multi_sensor import MultiSensor
from carried_objects.gimbaled_sensor import GimbaledSensor
from utils.transformations import get_rotation_matrix

def animate_gimbaled_system():
    # --- Animation & Simulation Settings ---
    gif_filename = "gimbal_animation.gif"
    temp_frame_dir = "temp_frames_gimbal"
    simulation_duration = 40
    dt = 0.5

    if os.path.exists(temp_frame_dir): shutil.rmtree(temp_frame_dir)
    os.makedirs(temp_frame_dir)
    
    # --- System Initialization ---
    drone = AdvancedDrone(
        position=[0, 0, -3000], velocity=[100, 0, 0], acceleration=[0, 0, 0],
        max_heading_change_rate=np.deg2rad(90), horsepower=10.0, max_static_thrust=500.0,
        max_circling_radius=1500, circling_radius_rate=20.0
    )
    multi_sensor = MultiSensor(installation_angles_deg=[0, -30, 0])
    
    # Add two different gimbaled sensors to the platform
    sensor1 = GimbaledSensor(
        installation_angles=[0,0,0], resolution=(640,480), hfov_deg=10, vfov_deg=10,
        max_gimbal_rate_dps=90, max_gimbal_yaw_deg=120, max_gimbal_pitch_deg=45
    )
    sensor2 = GimbaledSensor(
        installation_angles=[0,0,0], resolution=(640,480), hfov_deg=25, vfov_deg=25,
        max_gimbal_rate_dps=45, max_gimbal_yaw_deg=90, max_gimbal_pitch_deg=30
    )
    multi_sensor.add_sensor(sensor1)
    multi_sensor.add_sensor(sensor2)
    
    z_plane = 0.0
    target_pos = np.array([-2000., -2000., z_plane])

    # --- Simulation Loop ---
    filenames = []
    time_steps = np.arange(0, simulation_duration, dt)
    print("Generating frames...")
    for i, t in enumerate(time_steps):
        # Drone follows a waypoint for the first part, then circles
        waypoint = np.array([2500, 2500, -2800]) if t < simulation_duration / 2.5 else None
        drone.step(dt, waypoint)
        
        # Target moves linearly across the ground
        target_pos[0] += 100 * dt
        target_pos[1] += 50 * dt
        
        # Update the multi-sensor system
        multi_sensor.step(target_pos, drone.position, drone.attitude, dt)

        # --- Plotting ---
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot Drone and Target
        ax.scatter(drone.position[1], drone.position[0], -drone.position[2], c='black', marker='o', s=100, label='Drone')
        ax.scatter(target_pos[1], target_pos[0], -target_pos[2], c='red', marker='x', s=100, label='Target')

        # Plot footprints for each sensor
        colors = ['cyan', 'magenta']
        for s_idx, sensor in enumerate(multi_sensor.sensors):
            intersections = sensor.calculate_footprint(drone.position, drone.attitude, multi_sensor.installation_angles_rad, z_plane)
            plot_order = ["bottom_left", "bottom_right", "top_right", "top_left"]
            verts = [intersections[key] for key in plot_order if key in intersections]
            if len(verts) >= 3:
                plot_verts = [[(v[1], v[0], -v[2]) for v in verts]]
                ax.add_collection3d(Poly3DCollection(plot_verts, alpha=0.3, facecolors=colors[s_idx]))
                # Also plot the rays
                for v in verts:
                    ax.plot([drone.position[1], v[1]], [drone.position[0], v[0]], [-drone.position[2], -v[2]], color=colors[s_idx], linestyle=':', alpha=0.5)

        # Configure Axes and save
        ax.set_xlabel("East"); ax.set_ylabel("North"); ax.set_zlabel("Altitude")
        ax.set_title(f"Multi-Sensor Tracking (Time: {t:.1f}s)")
        ax.set_xlim(-4000, 4000); ax.set_ylim(-4000, 4000); ax.set_zlim(0, 3500)
        ax.view_init(elev=60, azim=-45)

        filename = f"{temp_frame_dir}/frame_{i:03d}.png"
        filenames.append(filename)
        plt.savefig(filename)
        plt.close(fig)
        print(f"  - Saved frame {i+1}/{len(time_steps)}")

    # --- Compile GIF ---
    print("\nCompiling GIF...")
    with imageio.get_writer(gif_filename, mode='I', duration=dt, loop=0) as writer:
        for filename in filenames:
            writer.append_data(imageio.imread(filename))
    
    shutil.rmtree(temp_frame_dir)
    print(f"\nSuccessfully created {gif_filename}")

if __name__ == "__main__":
    animate_gimbaled_system() 