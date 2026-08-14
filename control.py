import casadi as ca
import numpy as np

class MPCController:
    def __init__(self, N=20, dt=0.1):
        self.N = N
        self.dt = dt
        
        # parameters
        m = 1.05       # mass (kg)
        g = 9.81       # gravity (m/s^2)
        J = 0.0875     # moment of inertia (kg*m^2)
        L_cg = 0.5     # distance from CoM to gimbal (m)
        
        # objective
        Q = np.diag([10.0, 10.0, 1.0, 1.0, 10.0, 1.0]) # state 
        R = np.diag([0.1, 10.0])                       # control 
        S = np.diag([0.1 *100, 10.0 *10])              # control rate
        Q_terminal = Q * 10.0 *100                     # heavy terminal cost
        
        # state: px, pz, vx, vz, theta, omega
        x = ca.MX.sym('x', 6)
        px, pz, vx, vz, theta, omega = x[0], x[1], x[2], x[3], x[4], x[5]
        
        # control u: F, delta
        u = ca.MX.sym('u', 2)
        F, delta = u[0], u[1]
        
        # continuous dynamics
        px_dot = vx
        pz_dot = vz
        vx_dot = -F/m * ca.sin(theta + delta)
        vz_dot = F/m * ca.cos(theta + delta) - g
        theta_dot = omega
        omega_dot = -F/J * ca.sin(delta) * L_cg
        
        x_dot = ca.vertcat(px_dot, pz_dot, vx_dot, vz_dot, theta_dot, omega_dot)
        
        # state transition function
        f_cont = ca.Function('f_cont', [x, u], [x_dot])
        
        # RK4 discretization
        k1 = f_cont(x, u)
        k2 = f_cont(x + dt/2 * k1, u)
        k3 = f_cont(x + dt/2 * k2, u)
        k4 = f_cont(x + dt * k3, u)
        x_next = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        
        f_discrete = ca.Function('f_discrete', [x, u], [x_next])
        
        # problem formulation
        self.opti = ca.Opti()
        
        self.X = self.opti.variable(6, N+1)
        self.U = self.opti.variable(2, N)
        
        self.x_init = self.opti.parameter(6)
        self.x_ref = self.opti.parameter(6)
        
        cost = 0
        for k in range(N):
            # dynamics constraints
            self.opti.subject_to(self.X[:, k+1] == f_discrete(self.X[:, k], self.U[:, k]))
            
            # state constraints
            self.opti.subject_to(self.X[1, k] >= 0.0)  # pz >= 0
            
            # control constraints
            self.opti.subject_to(self.opti.bounded(0.0, self.U[0, k], 20.0))    # Thrust limits
            self.opti.subject_to(self.opti.bounded(-0.26, self.U[1, k], 0.26))  # Gimbal limits
            
            # objective function
            err = self.X[:, k] - self.x_ref
            cost += ca.mtimes([err.T, Q, err])
            cost += ca.mtimes([self.U[:, k].T, R, self.U[:, k]])
            
            # soft constraint for angle limits to prevent infeasibility
            """angle = self.X[4, k]
            angle_limit = np.pi/4
            slack = ca.fmax(0, ca.fabs(angle) - angle_limit)
            cost += 10000.0 * slack**2"""
            
            if k > 0:
                du = self.U[:, k] - self.U[:, k-1]
                cost += ca.mtimes([du.T, S, du])
                
        # terminal cost (soft constraint for end state)
        err_N = self.X[:, N] - self.x_ref
        cost += ca.mtimes([err_N.T, Q_terminal, err_N])
        
        # boundary conditions
        self.opti.subject_to(self.X[:, 0] == self.x_init)
        
        self.opti.minimize(cost)
        
        # solver options
        p_opts = {"expand": True}
        s_opts = {"max_iter": 100, "print_level": 0, "sb": "yes"}
        self.opti.solver("ipopt", p_opts, s_opts)

        # State for warm starting
        self.U_prev = np.zeros((2, N))
        self.U_prev[0, :] = m * g # initial guess hover thrust

    def solve(self, x_current, x_target):
        self.opti.set_value(self.x_init, x_current)
        self.opti.set_value(self.x_ref, x_target)
        
        # warm start
        self.opti.set_initial(self.U, self.U_prev)
        
        try:
            sol = self.opti.solve()
            U_res = sol.value(self.U)
            self.U_prev = U_res
            return U_res[0, 0], U_res[1, 0] # Return F and delta for timestep 0
        except Exception as e:
            # If optimization fails, reuse the previous initial command
            print("MPC Failed, using previous command")
            return self.U_prev[0, 0], self.U_prev[1, 0]
