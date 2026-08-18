import os
import numpy as np
import mujoco
import optuna
from control import MPCController

XML_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "2D_rocket.xml"))

def run_simulation(trial, alpha, beta, gamma, sim_time=10.0):
    m = mujoco.MjModel.from_xml_path(XML_PATH)
    d = mujoco.MjData(m)
    
    mpc = MPCController(N=20, dt=0.1, alpha=alpha, beta=beta, gamma=gamma)
    x_target = np.array([5.0, 10.0, 0.0, 0.0, 0.0, 0.0])
    
    physics_dt = m.opt.timestep   # 0.001s (1000Hz)
    mpc_dt = 0.1                  # 0.1s (10Hz)
    steps_per_mpc = int(round(mpc_dt / physics_dt))
    total_mpc_steps = int(round(sim_time / mpc_dt))
    total_physics_steps = int(round(sim_time / physics_dt))
    
    # 70% steady-state threshold (fixed based on total planned simulation horizon)
    N_ss = int(0.7 * total_mpc_steps)
    delta_max = 0.26
    
    current_thrust = 1.05 * 9.81  # hover thrust guess
    current_gimbal = 0.0
    
    cumulative_cost = 0.0
    mpc_step_idx = 0
    
    for step in range(total_physics_steps):
        if step % steps_per_mpc == 0:
            x_current = np.array([
                d.qpos[0], d.qpos[1], 
                d.qvel[0], d.qvel[1], 
                d.qpos[2], d.qvel[2]
            ])
            
            thrust_cmd, gimbal_cmd = mpc.solve(x_current, x_target)
            current_thrust = thrust_cmd
            current_gimbal = gimbal_cmd
            
            # --- Incremental Partial Fitness (ITAE) ---
            t_k = mpc_step_idx * mpc_dt
            step_itae = t_k * np.linalg.norm(x_current - x_target)
            
            # Saturation penalty only after 70% steady state threshold (t >= 7.0s)
            sat_penalty = 0.0
            if mpc_step_idx >= N_ss and abs(current_gimbal) >= (delta_max - 1e-4):
                sat_penalty = 100.0
                
            cumulative_cost += step_itae + sat_penalty
            
            # --- Optuna Pruning Hook ---
            if trial is not None:
                trial.report(cumulative_cost, step=mpc_step_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            mpc_step_idx += 1

        d.ctrl[0] = current_gimbal
        d.ctrl[1] = current_thrust
        
        mujoco.mj_step(m, d)
        
        # Early crash / tumbling / divergence guard
        if d.qpos[1] < -0.1 or abs(d.qpos[2]) > np.pi / 2 or np.isnan(d.qpos[0]):
            return 1e6  # Crash penalty

    return cumulative_cost


def objective(trial):
    alpha = trial.suggest_float('alpha', 0.01, 100, log=True)
    beta = trial.suggest_float('beta', 0.01, 100, log=True)
    gamma = trial.suggest_float('gamma', 0.01, 100, log=True)

    cost = run_simulation(trial, alpha, beta, gamma)
    return cost

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
