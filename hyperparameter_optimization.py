import os
import numpy as np
import mujoco
import optuna
from scipy.spatial.transform import Rotation as R
from sensor import Accelerometer, Gyroscope, GPS, Magnetometer
from multiplicative_ekf import MEKF
from control import MPCController
from util import quat_rotation_matrix

XML_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "2D_rocket.xml"))

def run_simulation(trial, alpha, beta, gamma, x_target, scenario_idx=0, cost_offset=0.0, sim_time=10.0):
    m = mujoco.MjModel.from_xml_path(XML_PATH)
    d = mujoco.MjData(m)
    
    mpc = MPCController(N=20, dt=0.1, alpha=alpha, beta=beta, gamma=gamma)
    
    physics_dt = m.opt.timestep   # 0.001s (1000Hz)
    mpc_dt = 0.1                  # 0.1s (10Hz)
    gps_frequency = 10.0
    steps_per_mpc = int(round(mpc_dt / physics_dt))
    steps_per_gps = int(round((1.0 / gps_frequency) / physics_dt))
    total_mpc_steps = int(round(sim_time / mpc_dt))
    total_physics_steps = int(round(sim_time / physics_dt))
    
    # 70% steady-state threshold
    N_ss = int(0.7 * total_mpc_steps)
    delta_max = 0.26
    
    current_thrust = -m.opt.gravity[2] * 1.05  # setpoint thrust
    current_gimbal_x = 0.0
    current_gimbal_y = 0.0
    
    # Sensors and MEKF setup
    accel = Accelerometer(3, 0.005, 0.001)
    gyro = Gyroscope(3, 0.005, 0.002)
    gps = GPS(3, 0.001)
    mag = Magnetometer(3, 0.01, 0.0)
    gyro_accumulator = []
    
    mujoco.mj_forward(m, d)
    
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
    
    cumulative_cost = 0.0
    mpc_step_idx = 0
    steps_to_gps = steps_per_gps

    for step in range(total_physics_steps):
        # Sensor sampling
        sensor_acc = accel.sample(d.sensordata[:3], physics_dt)
        sensor_gyro = gyro.sample(d.sensordata[3:6], physics_dt)
        sensor_mag = mag.sample(d.sensordata[6:9])
        gyro_accumulator.append(sensor_gyro.copy())
        
        # MEKF predict
        mekf.prediction(physics_dt, sensor_gyro.reshape(3, 1), sensor_acc.reshape(3, 1))
        
        steps_to_gps += 1
        if steps_to_gps >= steps_per_gps:
            gps_pos = gps.sample(np.array(d.qpos[0:3]).reshape(3, 1))
            mekf.correction(physics_dt * steps_to_gps, acc=sensor_acc.reshape(3, 1), mag=sensor_mag.reshape(3, 1), gps=gps_pos)
            steps_to_gps = 0
        
        if step % steps_per_mpc == 0:
            px, py, pz = mekf.pos[0], mekf.pos[1], mekf.pos[2]
            vx, vy, vz = mekf.vel[0], mekf.vel[1], mekf.vel[2]
            qw, qx, qy, qz = mekf.q[0], mekf.q[1], mekf.q[2], mekf.q[3]
            
            if len(gyro_accumulator) > 0:
                gyro_avg = np.mean(gyro_accumulator, axis=0)
                wx = gyro_avg[0] - mekf.gyr_b[0]
                wy = gyro_avg[1] - mekf.gyr_b[1]
                wz = gyro_avg[2] - mekf.gyr_b[2]
            else:
                R_b2w = quat_rotation_matrix([qw, qx, qy, qz])
                omega_body = R_b2w.T @ np.array([d.qvel[3], d.qvel[4], d.qvel[5]])
                wx, wy, wz = omega_body[0], omega_body[1], omega_body[2]
            
            gyro_accumulator.clear()
            
            x_current = [px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
            
            current_thrust, current_gimbal_x, current_gimbal_y = mpc.solve(x_current, x_target)
            
            # --- Incremental Partial Fitness (ITAE) evaluated against true state ---
            R_b2w_true = quat_rotation_matrix(d.qpos[3:7])
            omega_body_true = R_b2w_true.T @ np.array([d.qvel[3], d.qvel[4], d.qvel[5]])
            true_state = np.array([
                d.qpos[0], d.qpos[1], d.qpos[2],
                d.qvel[0], d.qvel[1], d.qvel[2],
                d.qpos[3], d.qpos[4], d.qpos[5], d.qpos[6],
                omega_body_true[0], omega_body_true[1], omega_body_true[2]
            ])
            t_k = mpc_step_idx * mpc_dt
            step_itae = t_k * np.linalg.norm(true_state - x_target)
            
            # Saturation penalty only after 70% steady state threshold (t >= 7.0s)
            sat_penalty = 0.0
            if mpc_step_idx >= N_ss and (abs(current_gimbal_x) >= (delta_max - 1e-4) or abs(current_gimbal_y) >= (delta_max - 1e-4)):
                sat_penalty = 100.0
                
            cumulative_cost += step_itae + sat_penalty
            
            # --- Optuna Pruning Hook across scenarios ---
            if trial is not None:
                global_step = scenario_idx * total_mpc_steps + mpc_step_idx
                trial.report(cost_offset + cumulative_cost, step=global_step)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            mpc_step_idx += 1

        d.ctrl[0] = current_gimbal_x
        d.ctrl[1] = current_gimbal_y
        d.ctrl[2] = current_thrust
        
        mujoco.mj_step(m, d)
        
        # Early crash guard (pz < 0 or large tilt)
        if d.qpos[2] < 0.0 or d.qpos[3] < 0.3 or np.isnan(d.qpos[0]):
            return 1e6  # Crash penalty

    return cumulative_cost


def objective(trial):
    alpha = trial.suggest_float('alpha', 0.01, 100, log=True)
    beta = trial.suggest_float('beta', 0.01, 100, log=True)
    gamma = trial.suggest_float('gamma', 0.01, 100, log=True)

    # Scenario 1: Straight up to z=10 and hover
    target_scenario_1 = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    cost_1 = run_simulation(trial, alpha, beta, gamma, x_target=target_scenario_1, scenario_idx=0, cost_offset=0.0)

    # Scenario 2: Go to x=5, y=5, z=10 and hover
    target_scenario_2 = np.array([5.0, 5.0, 10.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    cost_2 = run_simulation(trial, alpha, beta, gamma, x_target=target_scenario_2, scenario_idx=1, cost_offset=cost_1)

    return cost_1 + cost_2

if __name__ == "__main__":
    pruner = optuna.pruners.PercentilePruner(
        percentile=75.0, 
        n_startup_trials=10, 
        n_warmup_steps=20, 
        interval_steps=5
    )
    study = optuna.create_study(direction='minimize', pruner=pruner)
    study.optimize(objective, n_trials=30)

    print("Optimization finished!")
    print("Best Parameters:", study.best_params)
    print("Best Cost:", study.best_value)
