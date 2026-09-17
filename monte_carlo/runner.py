import os
import sys
import time
import multiprocessing as mp
import numpy as np
import xml.etree.ElementTree as ET
import json
import concurrent.futures

# Add parent dir to path so we can import from sim
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import mujoco as mj
from util import quat_to_euler, quat_rotation_matrix
from control import MPCController
from planner import MPCPlanner
from multiplicative_ekf import MEKF
from sensor import Accelerometer, Gyroscope, GPS, Magnetometer

XML_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "2D_rocket.xml"))

# Constants
ENABLE_FUEL_DEPLETION = True
ISP = 120.0
MIN_SLOSH_MASS = 0.005
MIN_FLUID_MASS = 0.005
G0 = 9.81

def run_trial(trial_id, seed):
    # Limit OpenMP threads for NumPy/SciPy/CasADi inside the worker process
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    np.random.seed(seed)
    
    # --- Sample Parameters ---
    init_pos = [
        np.random.uniform(-1.0, 1.0),
        np.random.uniform(-1.0, 1.0),
        np.random.uniform(8.0, 15.0)
    ]
    mass_multiplier = np.random.uniform(0.9, 1.1)
    gimbal_timeconst = np.random.uniform(0.02, 0.1)
    motor_dynprm = np.random.uniform(0.1, 0.4)
    wind_x = np.random.normal(0.0, 0.25)
    wind_y = np.random.normal(0.0, 0.25)
    
    # --- Setup XML and MuJoCo ---
    tree = ET.parse(XML_PATH)
    root = tree.getroot()
    for act in root.find('actuator'):
        if act.tag == 'position' and 'gimbal' in act.get('joint'):
            act.set('timeconst', str(gimbal_timeconst))
        elif act.tag == 'general' and act.get('site') == 'thrust_site':
            act.set('dynprm', str(motor_dynprm))
            
    xml_str = ET.tostring(root, encoding='unicode')
    m = mj.MjModel.from_xml_string(xml_str)
    d = mj.MjData(m)
    
    # Ground clearance
    rocket_body_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "rocket")
    geom_id = m.body_geomadr[rocket_body_id]
    ground_clearance = float(m.geom_size[geom_id, 1]) + 0.25

    # True mass vs Estimated mass
    true_initial_mass = float(np.sum(m.body_mass)) * mass_multiplier
    m.body_mass[:] = m.body_mass[:] * mass_multiplier
    mj.mj_setTotalmass(m, true_initial_mass)
    
    estimated_initial_mass = true_initial_mass / mass_multiplier # The nominal mass we feed the controller
    
    # Fuel depletion variables
    slosh_body_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "slosh_mass_xy")
    slosh_dummy_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "slosh_dummy_x")
    gimbal_pitch_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "gimbal_pitch")
    gimbal_yaw_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "gimbal_yaw")
    
    initial_slosh_mass = float(m.body_mass[slosh_body_id])
    initial_fluid_mass = 0.8 * true_initial_mass
    initial_solid_mass = 0.2 * true_initial_mass
    current_fluid_mass = initial_fluid_mass
    current_slosh_mass = initial_slosh_mass
    initial_tank_level = 0.2
    tank_radius = 0.08
    sphere_geom_id = m.body_geomadr[slosh_body_id]
    sphere_radius = float(m.geom_size[sphere_geom_id, 0]) if sphere_geom_id >= 0 else 0.08
    sphere_inertia_factor = 0.4 * (sphere_radius ** 2)

    other_bodies_mass = float(m.body_mass[slosh_dummy_id] + m.body_mass[gimbal_pitch_id] + m.body_mass[gimbal_yaw_id])

    # Initial State
    d.qpos[0:3] = init_pos
    d.qvel[:] = 0.0
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mj.mj_forward(m, d)
    
    # Controllers
    planner = MPCPlanner(
        N=60, alpha=1.0, beta=1.0, gamma=1.0, 
        ground_clearance=ground_clearance, 
        mass=estimated_initial_mass,
        dry_mass=0.20 * estimated_initial_mass + 0.005,
        isp=ISP
    )
    tracker = MPCController(
        N=15, dt=0.1, alpha=16.68, beta=0.58, gamma=4.48, 
        ground_clearance=ground_clearance, 
        mass=estimated_initial_mass
    )

    x_land = [0.0, 0.0, ground_clearance, 0.0, 0.0, 0.0]
    
    # State flags & timings
    physics_dt = m.opt.timestep
    last_guidance_time = -1.0
    last_tracker_time = -1.0
    t_final_current = 1e4
    latest_trajectory = None
    engine_cut = False
    
    current_thrust = -m.opt.gravity[2] * estimated_initial_mass
    current_gimbal_x = 0.0
    current_gimbal_y = 0.0
    
    # Metrics
    metrics = {
        "trial_id": trial_id,
        "seed": seed,
        "success": False,
        "landing_time": 0.0,
        "fuel_consumed": 0.0,
        "max_tracking_err_pos": 0.0,
        "max_tracking_err_vel": 0.0,
        "solver_failures": 0,
        "actuator_saturation_steps": 0,
        "final_tilt_deg": 0.0,
        "final_pos_error": 0.0,
        "final_vel_mag": 0.0,
        "sampled_params": {
            "init_pos": init_pos,
            "mass_multiplier": mass_multiplier,
            "gimbal_timeconst": gimbal_timeconst,
            "motor_dynprm": motor_dynprm,
            "wind_x": wind_x,
            "wind_y": wind_y
        }
    }

    # Setup Sensors and MEKF
    accel = Accelerometer(3, 0.005, 0.001)
    gyro = Gyroscope(3, 0.005, 0.002)
    gps = GPS(3, 0.001)
    mag = Magnetometer(3, 0.01, 0.0)

    p0 = np.array(d.qpos[0:3]).reshape(3, 1)
    v0 = np.array(d.qvel[0:3]).reshape(3, 1)
    q0 = np.array(d.qpos[3:7]).reshape(4, 1)
    
    P_initial = np.zeros((15, 15), dtype=float)
    np.fill_diagonal(P_initial[0:3, 0:3], 0.01)
    np.fill_diagonal(P_initial[3:6, 3:6], 1.0)
    np.fill_diagonal(P_initial[6:9, 6:9], 1.0)
    np.fill_diagonal(P_initial[9:12, 9:12], 0.001)
    np.fill_diagonal(P_initial[12:15, 12:15], 0.001)
    
    mekf = MEKF(q0, v0, p0, 0.05, 0.1, 0.001, 0.001, gps_cov=0.001, P_initial=P_initial)
    mag_world = np.array(m.opt.magnetic)
    mag_norm = np.linalg.norm(mag_world)
    if mag_norm > 1e-6:
        mekf.north_vector = (mag_world / mag_norm).reshape(3, 1)

    steps_to_gps = 0
    steps_per_gps = int((1.0 / 10.0) / physics_dt)
    
    gyro_accumulator = []
    
    # Helper to calculate mass
    estimated_current_mass = estimated_initial_mass

    max_sim_time = 30.0
    settle_time_after_land = 2.0
    landed_time = -1.0
    
    while d.time < max_sim_time:
        # Wind force
        d.xfrc_applied[rocket_body_id, 0] = wind_x
        d.xfrc_applied[rocket_body_id, 1] = wind_y

        # Sensor update
        sensor_acc = accel.sample(d.sensordata[:3], physics_dt)
        sensor_gyro = gyro.sample(d.sensordata[3:6], physics_dt)
        sensor_mag = mag.sample(d.sensordata[6:9])
        gyro_accumulator.append(sensor_gyro.copy())

        mekf.prediction(physics_dt, sensor_gyro.reshape(3, 1), sensor_acc.reshape(3, 1))

        # ---------------- Control Logic ----------------
        px, py, pz = float(mekf.pos[0]), float(mekf.pos[1]), float(mekf.pos[2])
        vx, vy, vz = float(mekf.vel[0]), float(mekf.vel[1]), float(mekf.vel[2])
        qw, qx, qy, qz = float(mekf.q[0]), float(mekf.q[1]), float(mekf.q[2]), float(mekf.q[3])
        
        if len(gyro_accumulator) > 0:
            gyro_avg = np.mean(gyro_accumulator, axis=0)
            wx = float(gyro_avg[0] - mekf.gyr_b[0])
            wy = float(gyro_avg[1] - mekf.gyr_b[1])
            wz = float(gyro_avg[2] - mekf.gyr_b[2])
        else:
            wx, wy, wz = 0.0, 0.0, 0.0
            
        x_current_6d = [px, py, pz, vx, vy, vz]
        x_current_13d = [px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]

        # 1. High-Level Guidance (2 Hz)
        if (d.time - last_guidance_time >= 0.5) and t_final_current > 3.0:
            t_elapsed = d.time - last_guidance_time if last_guidance_time > 0 else None
            x_init_plan = x_current_6d if t_elapsed is None else None
            
            traj, t_final_val = planner.solve(
                x_current=x_init_plan,
                x_target=x_land,
                t_elapsed=t_elapsed,
                t_plan_start=d.time,
                current_mass=estimated_current_mass if t_elapsed is None else None
            )
            
            if traj is not None:
                latest_trajectory = traj
                t_final_current = t_final_val
            else:
                metrics["solver_failures"] += 1
                
            last_guidance_time = d.time

        # 2. Low-Level Tracking (40 Hz)
        if d.time - last_tracker_time >= 0.025:
            gyro_accumulator.clear()
            
            if latest_trajectory is not None:
                traj_ref_13d = latest_trajectory.sample_13d(d.time, N_low=tracker.N, dt_low=tracker.dt)
                
                # Tracking error stats
                err_pos = np.linalg.norm(np.array(x_current_6d[:3]) - traj_ref_13d[:3, 0])
                err_vel = np.linalg.norm(np.array(x_current_6d[3:6]) - traj_ref_13d[3:6, 0])
                metrics["max_tracking_err_pos"] = max(metrics["max_tracking_err_pos"], err_pos)
                metrics["max_tracking_err_vel"] = max(metrics["max_tracking_err_vel"], err_vel)
                
            else:
                traj_ref_13d = np.zeros((13, tracker.N + 1))
                traj_ref_13d[:3, :] = np.array([px, py, pz])[:, None]
                traj_ref_13d[6, :] = 1.0 
                
            current_thrust, current_gimbal_x, current_gimbal_y = tracker.solve(
                x_current_13d, traj_ref_13d, current_mass=estimated_current_mass
            )
            
            # Saturation Check
            if current_thrust >= 19.9 or abs(current_gimbal_x) >= 0.25 or abs(current_gimbal_y) >= 0.25:
                metrics["actuator_saturation_steps"] += 1
                
            last_tracker_time = d.time
            
        # Touchdown & Landing Evaluation
        if pz < ground_clearance + 0.05:
            if not engine_cut:
                engine_cut = True
                landed_time = d.time
        
        # Actuation
        if engine_cut:
            d.ctrl[0] = 0.0
            d.ctrl[1] = 0.0
            d.ctrl[2] = 0.0
        else:
            d.ctrl[0] = current_gimbal_x
            d.ctrl[1] = current_gimbal_y
            d.ctrl[2] = current_thrust
            
        # ---------------- Step Dynamics ----------------
        mj.mj_step(m, d)
        
        # Propellant Depletion
        if ENABLE_FUEL_DEPLETION and current_fluid_mass > MIN_FLUID_MASS:
            actual_thrust = max(0.0, float(d.actuator_force[2]))
            dm = (actual_thrust / (ISP * G0)) * physics_dt
            current_fluid_mass = max(MIN_FLUID_MASS, current_fluid_mass - dm)
            
            # Update estimated mass for controller
            estimated_current_mass = estimated_initial_mass - (initial_fluid_mass - current_fluid_mass) * (estimated_initial_mass / true_initial_mass)
            
            # Update physical true mass properties
            true_current_mass = initial_solid_mass + current_fluid_mass
            h = max(1e-4, (current_fluid_mass / initial_fluid_mass) * initial_tank_level)
            current_slosh_mass = max(MIN_SLOSH_MASS, (0.4545 * (tank_radius / h) * np.tanh(1.8412 * h / tank_radius)) * current_fluid_mass)
            current_rigid_mass = true_current_mass - current_slosh_mass
            current_rocket_mass = max(0.01, current_rigid_mass - other_bodies_mass)

            m.body_mass[slosh_body_id] = current_slosh_mass
            m.body_inertia[slosh_body_id, :] = sphere_inertia_factor * current_slosh_mass

            m.body_mass[rocket_body_id] = current_rocket_mass
            m.body_inertia[rocket_body_id, 0] = (1.03 / 12.0) * current_rocket_mass
            m.body_inertia[rocket_body_id, 1] = (1.03 / 12.0) * current_rocket_mass
            m.body_inertia[rocket_body_id, 2] = 0.005 * current_rocket_mass

            m.body_subtreemass[slosh_body_id] = current_slosh_mass
            m.body_subtreemass[slosh_dummy_id] = m.body_mass[slosh_dummy_id] + current_slosh_mass
            m.body_subtreemass[1] = np.sum(m.body_mass[1:])
            m.body_subtreemass[0] = m.body_subtreemass[1]

        # GPS step
        steps_to_gps += 1
        if steps_to_gps >= steps_per_gps:
            gps_pos = gps.sample(np.array(d.qpos[0:3]).reshape(3, 1))
            mekf.correction(physics_dt * steps_to_gps, acc=sensor_acc.reshape(3, 1), mag=sensor_mag.reshape(3, 1), gps=gps_pos)
            steps_to_gps = 0
            
        if engine_cut and (d.time - landed_time > settle_time_after_land):
            break

    # Evaluate final metrics
    # The true tilt is the angle between the body Z-axis and the world Z-axis
    R_b2w = quat_rotation_matrix(d.qpos[3:7])
    # The Z-component of the body Z-axis in the world frame is R_b2w[2, 2]
    tilt_rad = np.arccos(np.clip(R_b2w[2, 2], -1.0, 1.0))
    tilt_deg = np.rad2deg(tilt_rad)
    
    pos_err = np.linalg.norm([d.qpos[0], d.qpos[1]])
    vel_mag = np.linalg.norm(d.qvel[0:3])
    
    metrics["final_tilt_deg"] = float(tilt_deg)
    metrics["final_pos_error"] = float(pos_err)
    metrics["final_vel_mag"] = float(vel_mag)
    metrics["landing_time"] = float(landed_time if engine_cut else d.time)
    metrics["fuel_consumed"] = float(initial_fluid_mass - current_fluid_mass)
    
    # Success Criteria: vertical attitude within a certain interval (e.g. 15 deg) after settling
    if engine_cut and tilt_deg < 15.0 and pos_err < 5.0:
        metrics["success"] = True

    return metrics


def main():
    num_trials = 50
    num_workers = min(4, mp.cpu_count())
    
    print(f"Starting Monte Carlo Simulation: {num_trials} trials with {num_workers} workers.")
    
    results = []
    
    t_start = time.time()
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(run_trial, i, 42 + i): i for i in range(num_trials)}
        for future in concurrent.futures.as_completed(futures):
            i = futures[future]
            try:
                res = future.result()
                results.append(res)
                succ = "SUCCESS" if res["success"] else "FAIL"
                print(f"Trial {i:02d} | {succ} | Time: {res['landing_time']:.1f}s | Tilt: {res['final_tilt_deg']:.1f}deg | Fuel: {res['fuel_consumed']:.3f}kg")
            except Exception as exc:
                print(f"Trial {i} generated an exception: {exc}")
                
    t_total = time.time() - t_start
    print(f"Completed {num_trials} trials in {t_total:.2f} seconds ({t_total/num_trials:.2f} s/trial)")
    
    # Aggregate and Save
    success_rate = sum(1 for r in results if r["success"]) / num_trials * 100
    print(f"Overall Success Rate: {success_rate:.1f}%")
    
    report_path = os.path.join(os.path.dirname(__file__), "results.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Saved detailed results to {report_path}")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()



# beauty of joseon revive eye serum
# topicals faded under eye mask