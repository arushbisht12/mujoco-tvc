import casadi as ca
import numpy as np

class MPCController:
    def __init__(self, N=20, dt=0.1, alpha=1.0, beta=1.0, gamma=1.0):
        self.N = N
        self.dt = dt
        
        # Physical parameters
        m = 1.05       # mass (kg)
        g = 9.81       # gravity (m/s^2)
        Jx = 0.0875    # moment of inertia x (kg*m^2)
        Jy = 0.0875    # moment of inertia y (kg*m^2)
        Jz = 0.00525   # moment of inertia z (kg*m^2)
        L_cg = 0.5     # distance from CoM to gimbal (m)
        
        # Base weight matrices
        # State: [px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
        # qx (tilt X) and qy (tilt Y) are actuated by the 2 gimbal axes
        # qz (yaw / spin around longitudinal Z axis) has zero torque authority (tau_z = 0) and zero weight
        Q_base = np.diag([
            15.0, 15.0, 40.0,  # px, py, pz
            5.0,  5.0,  15.0,  # vx, vy, vz
            0.0,  60.0, 60.0, 0.0, # qw, qx (tilt X), qy (tilt Y), qz (yaw unactuated = 0)
            3.0,  3.0,  0.0    # wx, wy, wz (wz unactuated = 0)
        ])
        
        # Control: [F, delta_x, delta_y]
        R_base = np.diag([0.05, 10.0, 10.0])
        
        # Control rate: [dF, d_delta_x, d_delta_y]
        S_base = np.diag([5.0, 50.0, 50.0])
        
        # Scaled weights
        Q = alpha * Q_base
        R = beta * R_base
        S = gamma * S_base
        Q_terminal = Q * 5.0
        
        # Symbolic variables
        # State x: [px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
        x = ca.MX.sym('x', 13)
        px, py, pz = x[0], x[1], x[2]
        vx, vy, vz = x[3], x[4], x[5]
        qw, qx, qy, qz = x[6], x[7], x[8], x[9]
        wx, wy, wz = x[10], x[11], x[12]
        
        # Control u: [F, delta_x, delta_y]
        u = ca.MX.sym('u', 3)
        F, delta_x, delta_y = u[0], u[1], u[2]
        
        # TVC Thrust vector in body frame matching MuJoCo kinematics
        # gimbal_y (hinge Y) tilts local thrust site Z toward +X in body
        # gimbal_x (hinge X) tilts local thrust site Z toward -Y in body
        fb_x = F * ca.sin(delta_y) * ca.cos(delta_x)
        fb_y = -F * ca.sin(delta_x)
        fb_z = F * ca.cos(delta_y) * ca.cos(delta_x)
        f_body = ca.vertcat(fb_x, fb_y, fb_z)
        
        # TVC Torques in body frame: r_thrust x f_body where r_thrust = [0, 0, -L_cg]
        tau_x = L_cg * fb_y
        tau_y = -L_cg * fb_x
        tau_z = 0.0
        
        # Polynomial rotation matrix from quaternion (body to world)
        R_b2w = ca.vertcat(
            ca.horzcat(1.0 - 2.0*(qy**2 + qz**2), 2.0*(qx*qy - qw*qz),       2.0*(qx*qz + qw*qy)),
            ca.horzcat(2.0*(qx*qy + qw*qz),       1.0 - 2.0*(qx**2 + qz**2), 2.0*(qy*qz - qw*qx)),
            ca.horzcat(2.0*(qx*qz - qw*qy),       2.0*(qy*qz + qw*qx),       1.0 - 2.0*(qx**2 + qy**2))
        )
        
        # World thrust and acceleration
        f_world = ca.mtimes(R_b2w, f_body)
        a_world = f_world / m + ca.vertcat(0, 0, -g)
        
        # Quaternion kinematics: q_dot = 0.5 * q (x) [0, omega]
        qw_dot = 0.5 * (-qx*wx - qy*wy - qz*wz)
        qx_dot = 0.5 * ( qw*wx + qy*wz - qz*wy)
        qy_dot = 0.5 * ( qw*wy - qx*wz + qz*wx)
        qz_dot = 0.5 * ( qw*wz + qx*wy - qy*wx)
        
        # Angular accelerations (Euler's equations for rigid body)
        wx_dot = (tau_x - (Jz - Jy) * wy * wz) / Jx
        wy_dot = (tau_y - (Jx - Jz) * wz * wx) / Jy
        wz_dot = (tau_z - (Jy - Jx) * wx * wy) / Jz
        
        # State derivatives
        x_dot = ca.vertcat(
            vx, vy, vz,
            a_world[0], a_world[1], a_world[2],
            qw_dot, qx_dot, qy_dot, qz_dot,
            wx_dot, wy_dot, wz_dot
        )
        
        # Continuous dynamics function
        f_cont = ca.Function('f_cont', [x, u], [x_dot])
        
        # RK4 discretization
        k1 = f_cont(x, u)
        k2 = f_cont(x + dt/2 * k1, u)
        k3 = f_cont(x + dt/2 * k2, u)
        k4 = f_cont(x + dt * k3, u)
        x_next = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        
        f_discrete = ca.Function('f_discrete', [x, u], [x_next])
        
        # Problem formulation
        self.opti = ca.Opti()
        
        self.X = self.opti.variable(13, N+1)
        self.U = self.opti.variable(3, N)
        
        self.x_init = self.opti.parameter(13)
        self.x_ref = self.opti.parameter(13)
        
        cost = 0
        for k in range(N):
            # Dynamics constraints
            self.opti.subject_to(self.X[:, k+1] == f_discrete(self.X[:, k], self.U[:, k]))
            
            # State constraints: pz >= 0
            self.opti.subject_to(self.X[2, k+1] >= 0.0)
            
            # Control constraints
            self.opti.subject_to(self.opti.bounded(0.0, self.U[0, k], 20.0))       # Thrust limits
            self.opti.subject_to(self.opti.bounded(-0.26, self.U[1, k], 0.26))     # Gimbal X limits (+-15 deg)
            self.opti.subject_to(self.opti.bounded(-0.26, self.U[2, k], 0.26))     # Gimbal Y limits (+-15 deg)
            
            # Stage cost
            err = self.X[:, k] - self.x_ref
            cost += ca.mtimes([err.T, Q, err])
            cost += ca.mtimes([self.U[:, k].T, R, self.U[:, k]])
            
            # Slew rate penalty
            if k > 0:
                du = self.U[:, k] - self.U[:, k-1]
                cost += ca.mtimes([du.T, S, du])
                
        # Terminal cost
        err_N = self.X[:, N] - self.x_ref
        cost += ca.mtimes([err_N.T, Q_terminal, err_N])
        
        # Boundary conditions
        self.opti.subject_to(self.X[:, 0] == self.x_init)
        
        self.opti.minimize(cost)
        
        # Tuned solver options for real-time MPC
        p_opts = {"expand": True, "print_time": False}
        s_opts = {
            "max_iter": 100,
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

        # State for warm starting
        self.U_prev = np.zeros((3, N))
        self.U_prev[0, :] = m * g # initial guess hover thrust
        self.X_prev = None

    def solve(self, x_current, x_target):
        self.opti.set_value(self.x_init, x_current)
        self.opti.set_value(self.x_ref, x_target)
        
        # Warm start both control and state trajectory
        self.opti.set_initial(self.U, self.U_prev)
        if self.X_prev is not None:
            self.opti.set_initial(self.X, self.X_prev)
        
        try:
            sol = self.opti.solve()
            U_res = sol.value(self.U)
            X_res = sol.value(self.X)
            # Shift horizon for warm starting next step
            self.U_prev = np.hstack([U_res[:, 1:], U_res[:, -1:]])
            self.X_prev = np.hstack([X_res[:, 1:], X_res[:, -1:]])
            # Return F, delta_x, delta_y for timestep 0
            return float(U_res[0, 0]), float(U_res[1, 0]), float(U_res[2, 0])
        except Exception as e:
            print(f"MPC Failed, using previous command: {e}")
            return float(self.U_prev[0, 0]), float(self.U_prev[1, 0]), float(self.U_prev[2, 0])
