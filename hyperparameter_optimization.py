import os
import time
import numpy as np
import mujoco
import optuna
from scipy.spatial.transform import Rotation as R

from sensor import Accelerometer, Gyroscope, GPS, Magnetometer
from multiplicative_ekf import MEKF
from control import MPCController
from planner import MPCPlanner
from util import quat_rotation_matrix

XML_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "2D_rocket.xml"))


def run_simulation(trial, alpha, beta, gamma, sim_time=18.0):
    """
    Runs a single simulation trial matching 2D_rocket.py:
    - 2-level architecture: High-level guidance planner + Low-level 40 Hz tracking MPC
    - MEKF state estimation with Accelerometer, Gyroscope, Magnetometer, and 10 Hz GPS
    - Tunes lower-level tracking MPC gains (alpha, beta, gamma)
    """
    m = mujoco.MjModel.from_xml_path(XML_PATH)
    d = mujoco.MjData(m)
    
    physics_dt = m.opt.timestep        # 0.001s (1000 Hz)
    tracker_dt = 0.025                 # 0.025s (40 Hz tracking MPC)
    gps_frequency = 10.0               # 10 Hz GPS
    
    steps_per_tracker = int(round(tracker_dt / physics_dt))
    steps_per_gps = int(round((1.0 / gps_frequency) / physics_dt))
    total_physics_steps = int(round(sim_time / physics_dt))
    
    # Ground clearance calculation matching 2D_rocket.py
    rocket_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "rocket")
    geom_id = m.body_geomadr[rocket_body_id]
    ground_clearance = float(m.geom_size[geom_id, 1]) + 0.25
    x_land = [0.0, 0.0, ground_clearance, 0.0, 0.0, 0.0]
    
    # Initialize Guidance Planner
    planner = MPCPlanner(
        N=60,
        alpha=1.0,
        beta=1.0,
        gamma=1.0,
        ground_clearance=ground_clearance
    )
    
    # Initialize Low-Level Tracking MPC with trial hyperparameters
    tracker = MPCController(
        N=15,
        dt=0.1,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        ground_clearance=ground_clearance
    )
    
    # Initial forward pass to populate state
    mujoco.mj_forward(m, d)
    
    # Solve initial guidance trajectory from rocket starting position
    x_init_6d = [float(d.qpos[0]), float(d.qpos[1]), float(d.qpos[2]), 0.0, 0.0, 0.0]
    latest_trajectory, t_final_current = planner.solve(x_init_6d, x_land)
    last_guidance_time = 0.0
    
    # Sensor and MEKF setup
    accel = Accelerometer(3, 0.005, 0.001)
    gyro = Gyroscope(3, 0.005, 0.002)
    gps = GPS(3, 0.001)
    mag = Magnetometer(3, 0.01, 0.0)
    gyro_accumulator = []
    
    p0 = np.array(d.qpos[0:3]).reshape(3, 1)
    v0 = np.array(d.qvel[0:3]).reshape(3, 1)
    q0 = np.array(d.qpos[3:7]).reshape(4, 1)
    
    gyro_cov = 0.05
    accel_cov = 0.1
    gyro_bias_cov = 0.001
    accel_bias_cov = 0.001
    gps_cov = 0.001
    
    P_initial = np.zeros((15, 15), dtype=float)
    np.fill_diagonal(P_initial[0:3, 0:3], 0.01)
    np.fill_diagonal(P_initial[3:6, 3:6], 1.0)
    np.fill_diagonal(P_initial[6:9, 6:9], 1.0)
    np.fill_diagonal(P_initial[9:12, 9:12], 0.001)
    np.fill_diagonal(P_initial[12:15, 12:15], 0.001)
    
    mag_world = np.array(m.opt.magnetic)
    mag_norm = np.linalg.norm(mag_world)
    
    mekf = MEKF(q0, v0, p0, gyro_cov, accel_cov, gyro_bias_cov, accel_bias_cov, gps_cov=gps_cov, P_initial=P_initial)
    if mag_norm > 1e-6:
        mekf.north_vector = (mag_world / mag_norm).reshape(3, 1)
    
    current_thrust = -m.opt.gravity[2] * 1.05
    current_gimbal_x = 0.0
    current_gimbal_y = 0.0
    prev_gimbal_x = 0.0
    prev_gimbal_y = 0.0
    
    steps_to_gps = 0
    tracker_step_idx = 0
    cumulative_cost = 0.0
    engine_cut = False
    delta_max = 0.26
    
    for step in range(total_physics_steps):
        # 1. Sensor sampling & MEKF prediction (1000 Hz)
        sensor_acc = accel.sample(d.sensordata[:3], physics_dt)
        sensor_gyro = gyro.sample(d.sensordata[3:6], physics_dt)
        sensor_mag = mag.sample(d.sensordata[6:9])
        gyro_accumulator.append(sensor_gyro.copy())
        
        mekf.prediction(physics_dt, sensor_gyro.reshape(3, 1), sensor_acc.reshape(3, 1))
        
        # 2. GPS correction (10 Hz)
        steps_to_gps += 1
        if steps_to_gps >= steps_per_gps:
            gps_pos = gps.sample(np.array(d.qpos[0:3]).reshape(3, 1))
            mekf.correction(physics_dt * steps_to_gps, acc=sensor_acc.reshape(3, 1), mag=sensor_mag.reshape(3, 1), gps=gps_pos)
            steps_to_gps = 0
            
        # 3. Guidance update (~2 Hz open loop replanning)
        if (d.time - last_guidance_time >= 0.5) and t_final_current > 3.0:
            t_elapsed = d.time - last_guidance_time
            latest_trajectory, t_final_current = planner.solve(
                x_current=None,
                x_target=x_land,
                t_elapsed=t_elapsed,
                t_plan_start=d.time
            )
            last_guidance_time = d.time
            
        # 4. Low-Level Tracking MPC update (40 Hz)
        if step % steps_per_tracker == 0:
            # Extract state feedback from MEKF
            px, py, pz = float(mekf.pos[0]), float(mekf.pos[1]), float(mekf.pos[2])
            vx, vy, vz = float(mekf.vel[0]), float(mekf.vel[1]), float(mekf.vel[2])
            qw, qx, qy, qz = float(mekf.q[0]), float(mekf.q[1]), float(mekf.q[2]), float(mekf.q[3])
            
            if len(gyro_accumulator) > 0:
                gyro_avg = np.mean(gyro_accumulator, axis=0)
                wx = float(gyro_avg[0] - mekf.gyr_b[0])
                wy = float(gyro_avg[1] - mekf.gyr_b[1])
                wz = float(gyro_avg[2] - mekf.gyr_b[2])
            else:
                R_b2w = quat_rotation_matrix([qw, qx, qy, qz])
                omega_body = R_b2w.T @ np.array([d.qvel[3], d.qvel[4], d.qvel[5]])
                wx, wy, wz = float(omega_body[0]), float(omega_body[1]), float(omega_body[2])
                
            gyro_accumulator.clear()
            x_current_13d = [px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
            
            # Sample continuous guidance trajectory preview
            traj_ref_13d = latest_trajectory.sample_13d(
                d.time,
                N_low=tracker.N,
                dt_low=tracker.dt
            )
            
            # Solve low-level tracking MPC
            prev_gimbal_x = current_gimbal_x
            prev_gimbal_y = current_gimbal_y
            current_thrust, current_gimbal_x, current_gimbal_y = tracker.solve(x_current_13d, traj_ref_13d)
            
            # --- Tracking Cost Evaluation against ground truth state ---
            p_true = np.array(d.qpos[0:3])
            v_true = np.array(d.qvel[0:3])
            q_true = np.array(d.qpos[3:7])
            
            p_ref = traj_ref_13d[:3, 0]
            v_ref = traj_ref_13d[3:6, 0]
            q_ref = traj_ref_13d[6:10, 0]
            
            pos_err = float(np.linalg.norm(p_true - p_ref))
            vel_err = float(np.linalg.norm(v_true - v_ref))
            # Quaternion attitude orientation alignment error: 1 - |<q_true, q_ref>|
            att_err = float(1.0 - abs(np.dot(q_true, q_ref)))
            
            # Actuation penalties: saturation and slew rate
            sat_penalty = 50.0 if (abs(current_gimbal_x) >= (delta_max - 1e-3) or abs(current_gimbal_y) >= (delta_max - 1e-3)) else 0.0
            slew_penalty = 5.0 * (abs(current_gimbal_x - prev_gimbal_x) + abs(current_gimbal_y - prev_gimbal_y))
            
            step_cost = (pos_err * 15.0 + vel_err * 5.0 + att_err * 30.0 + sat_penalty + slew_penalty) * tracker_dt
            cumulative_cost += step_cost
            
            # Touchdown and Landing Check
            if pz < ground_clearance + 0.05:
                vel_mag = float(np.linalg.norm([vx, vy, vz]))
                if vel_mag < 0.2:
                    engine_cut = True
                    # Successful soft landing: add terminal landing precision cost and finish trial
                    dist_to_pad = float(np.linalg.norm(p_true[:2]))
                    terminal_cost = dist_to_pad * 20.0 + vel_mag * 10.0
                    cumulative_cost += terminal_cost
                    break
            
            # Optuna Pruning Hook
            if trial is not None:
                trial.report(cumulative_cost, step=tracker_step_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()
                    
            tracker_step_idx += 1

        # 5. Actuation
        if engine_cut:
            d.ctrl[0] = 0.0
            d.ctrl[1] = 0.0
            d.ctrl[2] = 0.0
        else:
            d.ctrl[0] = current_gimbal_x
            d.ctrl[1] = current_gimbal_y
            d.ctrl[2] = current_thrust
            
        mujoco.mj_step(m, d)
        
        # Early crash guard: below ground, upside down (qw < 0.3), or NaN
        if d.qpos[2] < ground_clearance - 0.2 or d.qpos[3] < 0.3 or np.isnan(d.qpos[0]):
            return 1e6  # Crash penalty
            
    # If time runs out without touchdown, penalize altitude difference
    if not engine_cut:
        remaining_alt = max(0.0, float(d.qpos[2]) - ground_clearance)
        cumulative_cost += remaining_alt * 50.0

    return cumulative_cost


def objective(trial):
    """
    Optuna objective function for tuning the tracking MPC hyperparameters.
    Tunes alpha (tracking), beta (effort), and gamma (slew rate).
    """
    alpha = trial.suggest_float('alpha', 0.01, 100.0, log=True)
    beta = trial.suggest_float('beta', 0.01, 100.0, log=True)
    gamma = trial.suggest_float('gamma', 0.01, 100.0, log=True)

    # Run the nominal landing flight matching 2D_rocket.py
    cost = run_simulation(trial, alpha, beta, gamma)
    return cost


if __name__ == "__main__":
    pruner = optuna.pruners.PercentilePruner(
        percentile=75.0, 
        n_startup_trials=5, 
        n_warmup_steps=40,   # ~1.0 second of flight before pruning
        interval_steps=10
    )
    study = optuna.create_study(direction='minimize', pruner=pruner)
    study.optimize(objective, n_trials=30)

    print("=" * 60)
    print("Optimization finished!")
    print("Best Parameters:", study.best_params)
    print("Best Cost:", study.best_value)
    print("=" * 60)
