import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# 1. System Parameters
m = 1.05       # mass (kg)
g = 9.81       # gravity (m/s^2)
J = 0.0875     # moment of inertia (kg*m^2), estimated as 1/12 * m * L^2
L_cg = 0.5     # distance from CoM to gimbal (m)

# 2. Optimization Parameters
N = 50         # horizon
dt = 0.1       # timestep

# 3. Cost matrices
Q = np.diag([10.0, 10.0, 1.0, 1.0, 10.0, 1.0]) # State cost
R = np.diag([0.1, 10.0])                       # Control cost
S = np.diag([0.1, 10.0])                       # Control rate cost

def setup_casadi_opti():
    # 4. Dynamics Definition
    # States: px, pz, vx, vz, theta, omega
    x = ca.MX.sym('x', 6)
    px, pz, vx, vz, theta, omega = x[0], x[1], x[2], x[3], x[4], x[5]

    # Controls: F, delta
    u = ca.MX.sym('u', 2)
    F, delta = u[0], u[1]

    # Continuous dynamics
    px_dot = vx
    pz_dot = vz
    vx_dot = -F/m * ca.sin(theta + delta)
    vz_dot = F/m * ca.cos(theta + delta) - g
    theta_dot = omega
    omega_dot = -F/J * ca.sin(delta) * L_cg

    x_dot = ca.vertcat(px_dot, pz_dot, vx_dot, vz_dot, theta_dot, omega_dot)

    # Continuous function
    f_cont = ca.Function('f_cont', [x, u], [x_dot])

    # RK4 Discretization
    k1 = f_cont(x, u)
    k2 = f_cont(x + dt/2 * k1, u)
    k3 = f_cont(x + dt/2 * k2, u)
    k4 = f_cont(x + dt * k3, u)
    x_next = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    f_discrete = ca.Function('f_discrete', [x, u], [x_next])

    # 5. Opti Formulation
    opti = ca.Opti()

    X = opti.variable(6, N+1)
    U = opti.variable(2, N)

    # Target references
    x_start = [0.0, 5.0, 0.0, 0.0, 0.0, 0.0]  # Start at [0, 5]
    x_end = [5.0, 10.0, 0.0, 0.0, 0.0, 0.0]   # End at [5, 10]
    x_ref = np.array(x_end).reshape(-1, 1)

    # Cost function
    cost = 0

    for k in range(N):
        # Dynamics constraint
        opti.subject_to(X[:, k+1] == f_discrete(X[:, k], U[:, k]))
        
        # State constraints
        opti.subject_to(X[1, k] >= 0.0)  # pz >= 0
        
        # Control constraints
        opti.subject_to(opti.bounded(0.0, U[0, k], 20.0))    # Thrust limits
        opti.subject_to(opti.bounded(-0.26, U[1, k], 0.26))  # Gimbal limits
        
        # Running cost
        err = X[:, k] - x_ref
        cost += ca.mtimes([err.T, Q, err])
        cost += ca.mtimes([U[:, k].T, R, U[:, k]])
        
        if k > 0:
            du = U[:, k] - U[:, k-1]
            cost += ca.mtimes([du.T, S, du])

    # Boundary conditions
    opti.subject_to(X[:, 0] == x_start)
    opti.subject_to(X[:, N] == x_ref)

    # Initial guess to help solver
    for k in range(N):
        opti.set_initial(U[0, k], m*g) # hover thrust
        opti.set_initial(X[1, k], np.linspace(x_start[1], x_end[1], N+1)[k])
        opti.set_initial(X[0, k], np.linspace(x_start[0], x_end[0], N+1)[k])

    opti.minimize(cost)

    # Solver options
    p_opts = {"expand": True}
    s_opts = {"max_iter": 500, "print_level": 5}
    opti.solver("ipopt", p_opts, s_opts)

    return opti, X, U, N, dt, x_start, x_end

def main():
    opti, X, U, N, dt, x_start, x_end = setup_casadi_opti()

    print("Solving CasADi optimization...")
    try:
        sol = opti.solve()
        
        X_res = sol.value(X)
        U_res = sol.value(U)
        
        # 6. Plotting
        time = np.linspace(0, N*dt, N+1)
        
        plt.figure(figsize=(12, 10))
        
        # Trajectory
        plt.subplot(3, 2, 1)
        plt.plot(X_res[0, :], X_res[1, :], 'b-o', markersize=3)
        plt.plot(x_start[0], x_start[1], 'go', label='Start')
        plt.plot(x_end[0], x_end[1], 'ro', label='End')
        plt.xlabel('px (m)')
        plt.ylabel('pz (m)')
        plt.title('Trajectory in XZ plane')
        plt.legend()
        plt.grid(True)
        
        # Thrust
        plt.subplot(3, 2, 2)
        plt.plot(time[:-1], U_res[0, :], 'r-', drawstyle='steps-post')
        plt.axhline(20.0, color='k', linestyle='--')
        plt.axhline(0.0, color='k', linestyle='--')
        plt.xlabel('Time (s)')
        plt.ylabel('Thrust (N)')
        plt.title('Thrust Command')
        plt.grid(True)
        
        # Gimbal
        plt.subplot(3, 2, 3)
        plt.plot(time[:-1], U_res[1, :], 'g-', drawstyle='steps-post')
        plt.axhline(0.26, color='k', linestyle='--')
        plt.axhline(-0.26, color='k', linestyle='--')
        plt.xlabel('Time (s)')
        plt.ylabel('Gimbal Angle (rad)')
        plt.title('Gimbal Angle Command')
        plt.grid(True)
        
        # Theta
        plt.subplot(3, 2, 4)
        plt.plot(time, X_res[4, :], 'm-')
        plt.xlabel('Time (s)')
        plt.ylabel('Pitch Angle (rad)')
        plt.title('Pitch Angle over Time')
        plt.grid(True)
        
        # Velocities
        plt.subplot(3, 2, 5)
        plt.plot(time, X_res[2, :], label='vx')
        plt.plot(time, X_res[3, :], label='vz')
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity (m/s)')
        plt.title('Velocities')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('trajectory.png')
        print("Trajectory plotted and saved to trajectory.png")
        
    except Exception as e:
        print(f"Optimization failed: {e}")

if __name__ == "__main__":
    main()
