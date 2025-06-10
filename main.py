import numpy as np
import time

from drones.simple_drone import SimpleDrone
from drones.advanced_drone import AdvancedDrone
from carried_objects.sensor import Sensor
from carried_objects.rocket_launcher import RocketLauncher

def run_simulation():
    # --- Drone Initialization ---
    
    # Simple Drone
    simple_drone = SimpleDrone(
        position=[0, 0, -100], # Start at 100m altitude
        velocity=[0, 0, 0],
        acceleration=[0, 0, 0],
        max_heading_change_rate=np.deg2rad(90), # 90 deg/s
        horsepower=2.0,
        max_static_thrust=60.0 # Newtons
    )

    # Advanced Drone
    advanced_drone = AdvancedDrone(
        position=[50, 50, -100],
        velocity=[10, 0, 0],
        acceleration=[0, 0, 0],
        max_heading_change_rate=np.deg2rad(120),
        horsepower=4.0,
        max_static_thrust=100.0, # Newtons
        max_circling_radius=200,
        circling_radius_rate=2.0
    )
    
    # Add objects to the advanced drone
    sensor = Sensor(installation_angles=[np.deg2rad(90), 0, 0]) # Pointing right
    rocket_launcher = RocketLauncher(installation_angles=[0, np.deg2rad(-5), 0]) # Pointing slightly down
    advanced_drone.add_object(sensor)
    advanced_drone.add_object(rocket_launcher)

    drones = [simple_drone, advanced_drone]

    # --- Simulation Setup ---
    simulation_duration = 120  # seconds
    dt = 0.1  # time step
    
    waypoints_simple = [
        np.array([200, 200, -150]),
        np.array([-100, 100, -120]),
    ]
    current_waypoint_idx_simple = 0

    waypoints_advanced = [
        np.array([150, -150, -150]),
    ]
    current_waypoint_idx_advanced = 0

    # --- Simulation Loop ---
    print("Starting simulation...")
    for t in np.arange(0, simulation_duration, dt):
        print(f"\n--- Time: {t:.1f}s ---")

        # --- Update Simple Drone ---
        simple_wp = waypoints_simple[current_waypoint_idx_simple]
        simple_drone.step(dt, simple_wp)
        print(f"Simple Drone Position: {np.round(simple_drone.position, 2)}, "
              f"Velocity: {np.round(np.linalg.norm(simple_drone.velocity), 2)} m/s, "
              f"Attitude: {np.round(np.rad2deg(simple_drone.attitude), 1)} deg")
        
        if np.linalg.norm(simple_drone.position - simple_wp) < 10.0:
            print(f"Simple Drone reached waypoint {current_waypoint_idx_simple + 1}")
            if current_waypoint_idx_simple < len(waypoints_simple) - 1:
                current_waypoint_idx_simple += 1
        
        # --- Update Advanced Drone ---
        adv_wp = None
        if t < 40: # Fly to waypoint for the first 40 seconds
            adv_wp = waypoints_advanced[current_waypoint_idx_advanced]
        # After 40s, waypoint is None, so it should start circling
        
        advanced_drone.step(dt, adv_wp)
        print(f"Advanced Drone Position: {np.round(advanced_drone.position, 2)}, "
              f"Velocity: {np.round(np.linalg.norm(advanced_drone.velocity), 2)} m/s, "
              f"Circling: {advanced_drone.circling}")
        if advanced_drone.circling:
             print(f"  Circling Radius: {advanced_drone.circling_radius:.2f}m")


        time.sleep(dt/4) # To make it watchable

    print("\nSimulation finished.")

if __name__ == "__main__":
    run_simulation() 