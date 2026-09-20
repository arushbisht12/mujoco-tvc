import casadi as ca
import numpy as np
from scipy.interpolate import interp1d


class GuidanceTrajectory:
    """
    Encapsulates the continuous optimal guidance solution as cubic splines.
    This object is picklable and lightweight to transfer across multiprocessing queues.
    """
    def __init__(self, interp_X, interp_U, T_f_sol, t_plan_start=0.0):
        self.interp_X = interp_X
        self.interp_U = interp_U
        self.T_f_sol = float(T_f_sol)
        self.t_plan_start = float(t_plan_start)

    def sample_state_6d(self, t_rel):
        """Samples 6D state [px, py, pz, vx, vy, vz] at relative time t_rel from plan start."""
        t_clamped = float(np.clip(t_rel, 0.0, self.T_f_sol))
        return self.interp_X(t_clamped)[:6]

    def sample_state_7d(self, t_rel):
        """Samples 7D state [px, py, pz, vx, vy, vz, m] at relative time t_rel from plan start."""
        t_clamped = float(np.clip(t_rel, 0.0, self.T_f_sol))
        return self.interp_X(t_clamped)

    def sample_mass(self, t_rel):
        """Samples vehicle mass m at relative time t_rel from plan start."""
        t_clamped = float(np.clip(t_rel, 0.0, self.T_f_sol))
        return float(self.interp_X(t_clamped)[6])

    def sample_control_3d(self, t_rel):
        """Samples 3D control force [Fx, Fy, Fz] at relative time t_rel from plan start."""
        t_clamped = float(np.clip(t_rel, 0.0, self.T_f_sol))
        return self.interp_U(t_clamped)

    def sample_13d(self, t_sim, N_low=15, dt_low=0.1):
        """
        Samples the planned trajectory over the low-level preview window
        [t_sim, t_sim + N_low * dt_low] and synthesizes 13D reference states.
        
        Returns:
            traj_13d: 2D numpy array of shape (13, N_low + 1)
        """
        t_preview = (t_sim - self.t_plan_start) + np.arange(N_low + 1) * dt_low
        t_preview_clamped = np.clip(t_preview, 0.0, self.T_f_sol)
        
        # Interpolate 7D state (px, py, pz, vx, vy, vz, m)
        X_preview = self.interp_X(t_preview_clamped)  # (7, N_low + 1)
        
        # Interpolate 3D control force (Fx, Fy, Fz)
        U_preview = self.interp_U(t_preview_clamped)  # (3, N_low + 1)
        
        traj_13d = np.zeros((13, N_low + 1))
        # Position and velocity
        traj_13d[:6, :] = X_preview[:6, :]
        
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
            traj_13d[10:13, j] = 0.0
    
        return traj_13d


class MPCPlanner:
    def __init__(self, N=40, alpha=1.0, beta=1.0, gamma=1.0, ground_clearance=0.5, mass=1.05, approach_cone_deg=None, approach_cone_alt=None, dry_mass=None, isp=120.0):
        self.N = N
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.ground_clearance = ground_clearance
        self.approach_cone_deg = float(approach_cone_deg) if approach_cone_deg is not None else None
        self.approach_cone_alt = float(approach_cone_alt) if approach_cone_alt is not None else None

        # Physical parameters
        self.m = float(mass)       # initial/nominal mass (kg)
        self.g = 9.81              # gravity (m/s^2)
        self.isp = float(isp)      # rocket specific impulse (s)
        self.dry_mass = float(dry_mass) if dry_mass is not None else float(0.20 * self.m + 0.005)
        
        # Base weight matrices
        R_base = np.diag([1.0, 1.0, 0.05])
        S_base = np.diag([0.2, 0.2, 0.02])
        
        # Scaled weights
        w_T = 2.0 * alpha          # Time penalty weight
        w_fuel = 100.0 * beta      # Fuel consumption weight
        S = gamma * S_base         # Slew rate penalty weight
        
        # 3D Point mass continuous dynamics with mass depletion (7D state)
        # state x: [px, py, pz, vx, vy, vz, m]
        x = ca.MX.sym('x', 7)
        px, py, pz = x[0], x[1], x[2]
        vx, vy, vz = x[3], x[4], x[5]
        m_curr = x[6]
        
        # control u: [Fx, Fy, Fz]
        u = ca.MX.sym('u', 3)
        Fx, Fy, Fz = u[0], u[1], u[2]
        
        # Rocket specific impulse mass depletion: dm/dt = -||F|| / (Isp * g0)
        eps_f = 1e-6
        F_norm = ca.sqrt(Fx**2 + Fy**2 + Fz**2 + eps_f)
        m_dot = -F_norm / (self.isp * self.g)
        
        # State derivatives
        x_dot = ca.vertcat(
            vx, vy, vz,
            Fx / m_curr, Fy / m_curr, Fz / m_curr - self.g,
            m_dot
        )
        f_cont = ca.Function('f_cont', [x, u], [x_dot])
        
        # Problem formulation
        self.opti = ca.Opti()
        
        # Optimization variables: 7D state (pos, vel, mass)
        self.X = self.opti.variable(7, N + 1)
        self.U = self.opti.variable(3, N)
        self.T_f = self.opti.variable()  # Free final time
        
        if self.approach_cone_deg is not None:
            self.cone_slack = self.opti.variable(N)  # Soft slack for approach cone
            self.opti.subject_to(self.cone_slack >= 0.0)
        
        # Parameters
        self.x_init = self.opti.parameter(6)
        self.x_ref = self.opti.parameter(6)
        self.m_init = self.opti.parameter()
        self.opti.set_value(self.m_init, self.m)
        
        # Final time bounds
        self.opti.subject_to(self.opti.bounded(0.5, self.T_f, 30.0))
        dt = self.T_f / N
        
        # Pointing cone angle: max 25 degrees tilt
        tan_theta = float(np.tan(np.deg2rad(25.0)))
        
        # Objective: Time penalty + direct fuel consumed
        fuel_consumed = self.X[6, 0] - self.X[6, N]
        cost = w_T * self.T_f + w_fuel * fuel_consumed
        
        if self.approach_cone_deg is not None:
            w_cone_slack_quad = 100.0
            w_cone_slack_lin = 10.0
            cost += w_cone_slack_quad * ca.sum1(self.cone_slack**2) + w_cone_slack_lin * ca.sum1(self.cone_slack)
        
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
            self.opti.subject_to(self.X[2, k] >= self.ground_clearance)
            
            # Dry mass constraint (cannot burn more fuel than available)
            self.opti.subject_to(self.X[6, k] >= self.dry_mass)
            
            # Velocity-altitude glide slope (forces slower descent near the ground)
            self.opti.subject_to(self.X[5, k] >= -0.6 * (self.X[2, k] - self.ground_clearance) - 0.2)
            eps = 1e-6
            self.opti.subject_to(ca.sqrt(self.X[4, k]**2 + eps) <= 0.2 * self.X[2, k] + 0.1)
            self.opti.subject_to(ca.sqrt(self.X[3, k]**2 + eps) <= 0.2 * self.X[2, k] + 0.1)
            
            # Approach Cone Constraint (optional, with soft slack in squared form)
            if self.approach_cone_deg is not None:
                tan_cone = float(np.tan(np.deg2rad(self.approach_cone_deg)))
                err_xy = self.X[:2, k] - self.x_ref[:2]
                err_z = self.X[2, k] - self.x_ref[2]
                self.opti.subject_to(err_z >= 0.0)
                
                err_xy_sq = ca.sumsqr(err_xy)
                cone_bound_sq = (err_z * tan_cone)**2
                
                if self.approach_cone_alt is not None:
                    delta = 0.5  # smooth transition width (m)
                    sigma = 1.0 / (1.0 + ca.exp((err_z - self.approach_cone_alt) / delta))
                    self.opti.subject_to(sigma * (err_xy_sq - cone_bound_sq) <= self.cone_slack[k])
                else:
                    self.opti.subject_to(err_xy_sq - cone_bound_sq <= self.cone_slack[k])
            
            # Control constraints
            F_k = self.U[:, k]
            self.opti.subject_to(F_k[2] >= 0.5)                                    # Positive vertical thrust
            self.opti.subject_to(F_k[0]**2 + F_k[1]**2 + F_k[2]**2 <= 20.0**2)     # Max thrust (20 N)
            self.opti.subject_to(F_k[0]**2 + F_k[1]**2 <= (F_k[2] * tan_theta)**2) # Pointing cone
            
            # Lateral control regularizer
            cost += dt * 0.1 * (F_k[0]**2 + F_k[1]**2)
            
            # Slew rate penalty
            if k > 0:
                du = self.U[:, k] - self.U[:, k-1]
                cost += ca.mtimes([du.T, S, du])
                
        # Terminal altitude & dry mass constraints
        self.opti.subject_to(self.X[2, N] >= self.ground_clearance)
        self.opti.subject_to(self.X[6, N] >= self.dry_mass)
        
        # Boundary conditions
        self.opti.subject_to(self.X[:6, 0] == self.x_init)
        self.opti.subject_to(self.X[6, 0] == self.m_init)
        self.opti.subject_to(self.X[:6, N] == self.x_ref)  # Land exactly at target state
        
        self.opti.minimize(cost)
        
        # Tuned IPOPT solver options
        p_opts = {"expand": True, "print_time": False}
        s_opts = {
            "max_iter": 200,
            "tol": 1e-1,
            "dual_inf_tol": 1e-2,
            "constr_viol_tol": 1e-1,
            "compl_inf_tol": 1e-1,
            "acceptable_tol": 1e-1,
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
        self.cone_slack_sol = None
        self.t_grid_sol = None
        self.has_solution = False
        self.last_valid_plan_start = 0.0
        self.x_ref_val = None

    def estimate_time_to_go(self, x_current, x_target):
        """Analytical time-to-go estimate used ONLY for initial warm starting without prior plan."""
        p_curr = np.asarray(x_current[:3], dtype=float)
        v_curr = np.asarray(x_current[3:6], dtype=float)
        p_tgt = np.asarray(x_target[:3], dtype=float)
        
        dist_3d = np.linalg.norm(p_curr - p_tgt)
        vel_mag = np.linalg.norm(v_curr)
        
        # Estimate based on average trajectory velocity / deceleration capability
        t_est = 2.0 * dist_3d / (vel_mag + 4.0)
        
        # Vertical descent estimate
        pz = max(0.1, p_curr[2] - p_tgt[2])
        vz = abs(v_curr[2])
        t_z = 2.0 * pz / (vz + 3.5)
        
        t_final_est = max(t_est, t_z)
        return float(np.clip(t_final_est, 1.0, 18.0))

    def sample_previous_solution(self, t_rel):
        """Samples the 6D state from the previous optimal trajectory at relative time t_rel."""
        if not self.has_solution or self.X_sol is None:
            return None
        t_clamped = float(np.clip(t_rel, 0.0, self.T_f_sol))
        interp_X = interp1d(self.t_grid_sol, self.X_sol, axis=1, kind='cubic', fill_value="extrapolate")
        return interp_X(t_clamped)[:6]

    def sample_previous_solution_7d(self, t_rel):
        """Samples the 7D state [px, py, pz, vx, vy, vz, m] from previous optimal trajectory."""
        if not self.has_solution or self.X_sol is None:
            return None
        t_clamped = float(np.clip(t_rel, 0.0, self.T_f_sol))
        interp_X = interp1d(self.t_grid_sol, self.X_sol, axis=1, kind='cubic', fill_value="extrapolate")
        return interp_X(t_clamped)

    def _build_guidance_trajectory(self, t_plan_start=0.0):
        """Constructs and returns a GuidanceTrajectory object from current solution arrays."""
        interp_X = interp1d(self.t_grid_sol, self.X_sol, axis=1, kind='cubic', fill_value="extrapolate")
        t_u_grid = self.t_grid_sol[:-1]
        interp_U = interp1d(t_u_grid, self.U_sol, axis=1, kind='linear', fill_value="extrapolate")
        return GuidanceTrajectory(interp_X, interp_U, self.T_f_sol, t_plan_start=t_plan_start)

    def solve(self, x_current, x_target, t_elapsed=None, t_plan_start=0.0, current_mass=None):
        """
        Solves the free-final-time point-mass trajectory optimization problem with 7D state (including mass).
        Maintains strictly open-loop mass progression during replanning (no external mass feedback).
        
        Args:
            x_current:   Current 6D state feedback [px, py, pz, vx, vy, vz].
                         If None and warm starting, samples from previous optimal trajectory.
            x_target:    Target landing 6D state
            t_elapsed:   Time elapsed (T_s,p) since previous plan start. If provided and a previous
                         solution exists, initial state is constrained to x*(T_s,p) from previous optimal
                         trajectory and horizon is shortened.
            t_plan_start: Simulation timestamp when this plan was initiated.
            current_mass: Real-time vehicle mass (kg) used ONLY on initial solve at t=0.
        Returns:
            trajectory: GuidanceTrajectory object with continuous interpolation functions and sampler
            T_f_sol:    float, optimal landing duration
        """
        x_tgt_arr = np.asarray(x_target, dtype=float).flatten()[:6]
        self.x_ref_val = x_tgt_arr
        self.opti.set_value(self.x_ref, x_tgt_arr)
        
        # Warm start initial guesses
        if self.has_solution and t_elapsed is not None:
            # 1. Strictly Open-Loop: sample 7D state (including mass) from previous step's optimal trajectory
            x_prev_7d = self.sample_previous_solution_7d(t_elapsed)
            self.opti.set_value(self.x_init, x_prev_7d[:6])
            self.opti.set_value(self.m_init, float(x_prev_7d[6]))  # Open-loop mass handoff
            
            # 2. Shorten the active optimization horizon: warm start with T_f - T_s,p
            t_warm = max(0.5, self.T_f_sol - float(t_elapsed))
            self.opti.set_initial(self.T_f, t_warm)
            
            # Warm start state and control trajectories from previous solution
            t_interp = np.linspace(float(t_elapsed), self.T_f_sol, self.N + 1)
            interp_X = interp1d(self.t_grid_sol, self.X_sol, axis=1, kind='cubic', fill_value="extrapolate")
            X_warm = interp_X(t_interp)
            self.opti.set_initial(self.X, X_warm)
            
            t_u_grid = self.t_grid_sol[:-1]
            interp_U = interp1d(t_u_grid, self.U_sol, axis=1, kind='linear', fill_value="extrapolate")
            t_u_interp = np.linspace(float(t_elapsed), self.T_f_sol, self.N)
            self.opti.set_initial(self.U, interp_U(t_u_interp))
            if self.approach_cone_deg is not None:
                self.opti.set_initial(self.cone_slack, 0.0)
        else:   
            # Initial solve without previous solution: construct full horizon with x_current
            m_val = float(current_mass) if current_mass is not None else self.m
            self.opti.set_value(self.m_init, m_val)
            
            x_curr_arr = np.asarray(x_current, dtype=float).flatten()[:6]
            self.opti.set_value(self.x_init, x_curr_arr)

            t_init = self.estimate_time_to_go(x_curr_arr, x_tgt_arr)
            self.opti.set_initial(self.T_f, t_init)
            
            # Linear state interpolation for position and velocity (first 6 states)
            X_init = np.zeros((7, self.N + 1))
            for i in range(6):
                X_init[i, :] = np.linspace(x_curr_arr[i], x_tgt_arr[i], self.N + 1)
            
            # Linear mass estimate across horizon
            est_burn = min(0.4 * (m_val - self.dry_mass), (m_val / self.isp) * t_init)
            X_init[6, :] = np.linspace(m_val, max(self.dry_mass, m_val - est_burn), self.N + 1)
            self.opti.set_initial(self.X, X_init)
            
            # Hover thrust guess
            U_init = np.zeros((3, self.N))
            U_init[2, :] = m_val * self.g
            self.opti.set_initial(self.U, U_init)
            if self.approach_cone_deg is not None:
                self.opti.set_initial(self.cone_slack, 0.0)
            
        try:
            sol = self.opti.solve()
            self.T_f_sol = float(sol.value(self.T_f))
            self.X_sol = np.array(sol.value(self.X))
            self.U_sol = np.array(sol.value(self.U))
            if self.approach_cone_deg is not None:
                self.cone_slack_sol = np.array(sol.value(self.cone_slack))
            self.t_grid_sol = np.linspace(0.0, self.T_f_sol, self.N + 1)
            self.has_solution = True
            self.last_valid_plan_start = float(t_plan_start)
            return self._build_guidance_trajectory(t_plan_start=t_plan_start), self.T_f_sol
        except Exception as e:
            print(f"Guidance MPC failed to solve: {e}")
            if self.has_solution:
                # Do NOT overwrite time origin of the previous valid plan!
                return None, self.T_f_sol
            else:
                # Initial solve fallback: create analytical linear profile
                t_init = self.estimate_time_to_go(x_curr_arr, x_tgt_arr)
                self.T_f_sol = t_init
                self.t_grid_sol = np.linspace(0.0, t_init, self.N + 1)
                self.X_sol = np.zeros((7, self.N + 1))
                for i in range(6):
                    self.X_sol[i, :] = np.linspace(x_curr_arr[i], x_tgt_arr[i], self.N + 1)
                self.X_sol[6, :] = m_val
                self.U_sol = np.zeros((3, self.N))
                self.U_sol[2, :] = m_val * self.g
                self.has_solution = True
                self.last_valid_plan_start = float(t_plan_start)
                return self._build_guidance_trajectory(t_plan_start=t_plan_start), self.T_f_sol

    def get_reference_trajectory(self, t_sim, t_plan_start, N_low=15, dt_low=0.1):
        """
        Samples the latest planned trajectory over the low-level preview window.
        Provided for backwards compatibility; returns 13D reference matrix (13, N_low + 1).
        """
        if not self.has_solution or self.X_sol is None:
            traj_13d = np.zeros((13, N_low + 1))
            traj_13d[6, :] = 1.0  # qw = 1.0 (upright)
            return traj_13d
        traj = self._build_guidance_trajectory(t_plan_start=t_plan_start)
        return traj.sample_13d(t_sim, N_low=N_low, dt_low=dt_low)


def guidance_worker_loop(req_queue, res_queue, planner_config):
    """
    Dedicated worker process loop for the Guidance MPC solver.
    Instantiates the CasADi/IPOPT planner in its own isolated process space.
    """
    planner = MPCPlanner(**planner_config)
    while True:
        try:
            req = req_queue.get()
            if req is None:
                break
            if len(req) == 5:
                t_req, x_init, x_target, t_elapsed, current_mass = req
            else:
                t_req, x_init, x_target, t_elapsed = req
                current_mass = None
            traj, t_final_val = planner.solve(
                x_current=x_init,
                x_target=x_target,
                t_elapsed=t_elapsed,
                t_plan_start=t_req,
                current_mass=current_mass
            )
            # Only put valid, non-None trajectory results to prevent resetting plan time origin
            if traj is not None:
                res_queue.put({
                    "t_req": t_req,
                    "trajectory": traj,
                    "t_final": t_final_val
                })
        except Exception as e:
            print(f"[Guidance Worker Error] {e}")