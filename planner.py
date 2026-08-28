import casadi as ca
import numpy as np
from scipy.interpolate import interp1d

class MPCPlanner:
    def __init__(self, N=30, alpha=1.0, beta=1.0, gamma=1.0):
        self.N = N
        
        # Physical parameters
        self.m = 1.05       # mass (kg)
        self.g = 9.81       # gravity (m/s^2)
        
        # Base weight matrices
        # Control force: [Fx, Fy, Fz]
        R_base = np.diag([1.0, 1.0, 0.05])
        # Control rate: [dFx, dFy, dFz]
        S_base = np.diag([0.2, 0.2, 0.02])
        
        # Scaled weights
        w_T = 2.0 * alpha          # Time penalty weight
        R = beta * R_base          # Fuel / control effort weight
        S = gamma * S_base         # Slew rate penalty weight
        
        # 3D Point mass continuous dynamics
        # state x: [px, py, pz, vx, vy, vz]
        x = ca.MX.sym('x', 6)
        px, py, pz = x[0], x[1], x[2]
        vx, vy, vz = x[3], x[4], x[5]
        
        # control u: [Fx, Fy, Fz]
        u = ca.MX.sym('u', 3)
        Fx, Fy, Fz = u[0], u[1], u[2]
        
        # State derivatives
        x_dot = ca.vertcat(
            vx, vy, vz,
            Fx / self.m, Fy / self.m, Fz / self.m - self.g
        )
        f_cont = ca.Function('f_cont', [x, u], [x_dot])
        
        # Problem formulation
        self.opti = ca.Opti()
        
        # Optimization variables
        self.X = self.opti.variable(6, N + 1)
        self.U = self.opti.variable(3, N)
        self.T_f = self.opti.variable()  # Free final time
        
        # Parameters
        self.x_init = self.opti.parameter(6)
        self.x_ref = self.opti.parameter(6)
        
        # Variable time step dt = T_f / N
        dt = self.T_f / N
        
        # Final time bounds
        self.opti.subject_to(self.opti.bounded(0.5, self.T_f, 30.0))
        
        # Pointing cone angle: max 25 degrees tilt
        tan_theta = float(np.tan(np.deg2rad(25.0)))
        
        cost = w_T * self.T_f
        
        for k in range(N):
            # RK4 Integration with symbolic dt = T_f / N
            k1 = f_cont(self.X[:, k], self.U[:, k])
            k2 = f_cont(self.X[:, k] + dt/2 * k1, self.U[:, k])
            k3 = f_cont(self.X[:, k] + dt/2 * k2, self.U[:, k])
            k4 = f_cont(self.X[:, k] + dt * k3, self.U[:, k])
            x_next = self.X[:, k] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            
            # Dynamics constraint
            self.opti.subject_to(self.X[:, k+1] == x_next)
            
            # Altitude constraint
            self.opti.subject_to(self.X[2, k] >= 0.0)
            
            # Control constraints
            F_k = self.U[:, k]
            self.opti.subject_to(F_k[2] >= 0.5)                                    # Positive vertical thrust
            self.opti.subject_to(F_k[0]**2 + F_k[1]**2 + F_k[2]**2 <= 20.0**2)     # Max thrust (20 N)
            self.opti.subject_to(F_k[0]**2 + F_k[1]**2 <= (F_k[2] * tan_theta)**2) # Pointing cone
            
            # Control effort cost (energy/fuel integral: sum dt * u^T R u)
            cost += dt * ca.mtimes([F_k.T, R, F_k])
            
            # Slew rate penalty
            if k > 0:
                du = self.U[:, k] - self.U[:, k-1]
                cost += ca.mtimes([du.T, S, du])
                
        # Terminal altitude constraint
        self.opti.subject_to(self.X[2, N] >= 0.0)
        
        # Boundary conditions
        self.opti.subject_to(self.X[:, 0] == self.x_init)
        self.opti.subject_to(self.X[:, N] == self.x_ref)  # Land exactly at target state
        
        self.opti.minimize(cost)
        
        # Tuned IPOPT solver options
        p_opts = {"expand": True, "print_time": False}
        s_opts = {
            "max_iter": 150,
            "tol": 1e-3,
            "dual_inf_tol": 1e-2,
            "constr_viol_tol": 1e-3,
            "compl_inf_tol": 1e-3,
            "acceptable_tol": 1e-2,
            "acceptable_iter": 5,
            "warm_start_init_point": "yes",
            "warm_start_bound_push": 1e-4,
            "warm_start_mult_bound_push": 1e-4,
            "print_level": 0,
            "sb": "yes"
        }
        self.opti.solver("ipopt", p_opts, s_opts)
        
        # Warm start state
        self.T_f_sol = 5.0
        self.X_sol = None
        self.U_sol = None
        self.t_grid_sol = None
        self.has_solution = False

    def estimate_time_to_go(self, x_current, x_target):
        """Analytical time-to-go estimate used for initial warm starting."""
        p_curr = np.asarray(x_current[:3], dtype=float)
        v_curr = np.asarray(x_current[3:6], dtype=float)
        p_tgt = np.asarray(x_target[:3], dtype=float)
        
        dist_3d = np.linalg.norm(p_curr - p_tgt)
        vel_mag = np.linalg.norm(v_curr)
        
        # Estimate based on average deceleration
        t_est = 2.0 * dist_3d / (vel_mag + 1.5)
        
        # Vertical descent estimate
        pz = max(0.1, p_curr[2] - p_tgt[2])
        vz = abs(v_curr[2])
        t_z = 2.0 * pz / (vz + 1.0)
        
        t_final_est = max(t_est, t_z)
        return float(np.clip(t_final_est, 1.0, 25.0))

    def solve(self, x_current, x_target):
        """
        Solves the free final time guidance NLP.
        Returns:
            t_grid: 1D numpy array of shape (N+1,) with timestamps from 0 to T_f
            X_sol:  2D numpy array of shape (6, N+1)
            U_sol:  2D numpy array of shape (3, N)
            T_f_sol: float, optimal landing duration
        """
        x_curr_arr = np.asarray(x_current, dtype=float).flatten()[:6]
        x_tgt_arr = np.asarray(x_target, dtype=float).flatten()[:6]
        
        self.opti.set_value(self.x_init, x_curr_arr)
        self.opti.set_value(self.x_ref, x_tgt_arr)
        
        # Warm start initial guesses
        if not self.has_solution:
            t_init = self.estimate_time_to_go(x_curr_arr, x_tgt_arr)
            self.opti.set_initial(self.T_f, t_init)
            
            # Linear state interpolation
            X_init = np.zeros((6, self.N + 1))
            for i in range(6):
                X_init[i, :] = np.linspace(x_curr_arr[i], x_tgt_arr[i], self.N + 1)
            self.opti.set_initial(self.X, X_init)
            
            # Hover thrust guess
            U_init = np.zeros((3, self.N))
            U_init[2, :] = self.m * self.g
            self.opti.set_initial(self.U, U_init)
        else:
            # Warm start from previous solution
            self.opti.set_initial(self.T_f, self.T_f_sol)
            self.opti.set_initial(self.X, self.X_sol)
            self.opti.set_initial(self.U, self.U_sol)
            
        try:
            sol = self.opti.solve()
            self.T_f_sol = float(sol.value(self.T_f))
            self.X_sol = np.array(sol.value(self.X))
            self.U_sol = np.array(sol.value(self.U))
            self.t_grid_sol = np.linspace(0.0, self.T_f_sol, self.N + 1)
            self.has_solution = True
            return self.t_grid_sol, self.X_sol, self.U_sol, self.T_f_sol
        except Exception as e:
            print(f"Guidance MPC failed to solve: {e}")
            if not self.has_solution:
                # Fallback: create analytical linear profile
                t_init = self.estimate_time_to_go(x_curr_arr, x_tgt_arr)
                self.T_f_sol = t_init
                self.t_grid_sol = np.linspace(0.0, t_init, self.N + 1)
                self.X_sol = np.zeros((6, self.N + 1))
                for i in range(6):
                    self.X_sol[i, :] = np.linspace(x_curr_arr[i], x_tgt_arr[i], self.N + 1)
                self.U_sol = np.zeros((3, self.N))
                self.U_sol[2, :] = self.m * self.g
            return self.t_grid_sol, self.X_sol, self.U_sol, self.T_f_sol

    def get_reference_trajectory(self, t_sim, t_plan_start, N_low=15, dt_low=0.1):
        """
        Samples the latest planned trajectory over the low-level preview window
        [t_sim, t_sim + N_low * dt_low] and synthesizes 13D reference states.
        
        Returns:
            traj_13d: 2D numpy array of shape (13, N_low + 1)
        """
        if not self.has_solution or self.X_sol is None:
            # Return upright hover default if no plan exists
            traj_13d = np.zeros((13, N_low + 1))
            traj_13d[6, :] = 1.0  # qw = 1.0 (upright)
            return traj_13d
            
        t_preview = t_sim - t_plan_start + np.arange(N_low + 1) * dt_low
        t_preview_clamped = np.clip(t_preview, 0.0, self.T_f_sol)
        
        # Interpolate 6D state (px, py, pz, vx, vy, vz)
        interp_X = interp1d(self.t_grid_sol, self.X_sol, axis=1, kind='linear', fill_value="extrapolate")
        X_preview = interp_X(t_preview_clamped)  # (6, N_low + 1)
        
        # Interpolate 3D control force (Fx, Fy, Fz)
        t_u_grid = self.t_grid_sol[:-1]
        interp_U = interp1d(t_u_grid, self.U_sol, axis=1, kind='linear', fill_value="extrapolate")
        U_preview = interp_U(t_preview_clamped)  # (3, N_low + 1)
        
        traj_13d = np.zeros((13, N_low + 1))
        # Position and velocity
        traj_13d[:6, :] = X_preview
        
        # Synthesize attitude quaternion from thrust direction
        for j in range(N_low + 1):
            F = U_preview[:, j]
            f_norm = np.linalg.norm(F)
            if f_norm > 1e-3:
                u_dir = F / f_norm  # Unit thrust vector in world frame
            else:
                u_dir = np.array([0.0, 0.0, 1.0])
                
            # Rocket longitudinal axis is +Z in body frame.
            # We want body +Z to align with desired thrust direction u_dir.
            # Axis of rotation: a = z_hat x u_dir = [-u_y, u_x, 0]
            z_hat = np.array([0.0, 0.0, 1.0])
            u_z = np.clip(u_dir[2], -1.0, 1.0)
            
            if u_z > 0.9999:
                # Perfectly upright
                q_ref = np.array([1.0, 0.0, 0.0, 0.0])
            elif u_z < -0.9999:
                # Inverted (tilt 180 deg around X)
                q_ref = np.array([0.0, 1.0, 0.0, 0.0])
            else:
                angle = np.arccos(u_z)
                axis = np.array([-u_dir[1], u_dir[0], 0.0])
                axis_norm = np.linalg.norm(axis)
                if axis_norm > 1e-6:
                    axis = axis / axis_norm
                    half_angle = angle / 2.0
                    q_ref = np.array([
                        np.cos(half_angle),
                        axis[0] * np.sin(half_angle),
                        axis[1] * np.sin(half_angle),
                        axis[2] * np.sin(half_angle)
                    ])
                else:
                    q_ref = np.array([1.0, 0.0, 0.0, 0.0])
                    
            traj_13d[6:10, j] = q_ref
            # Angular velocity reference omega = [0, 0, 0]
            traj_13d[10:13, j] = 0.0
            
        return traj_13d