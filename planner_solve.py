import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from planner import MPCPlanner


def main():
    # 1. Configuration matching 2D_rocket simulation
    ground_clearance = 0.75  # geom_size (0.5) + 0.25 offset
    N_horizon = 60
    
    planner = MPCPlanner(
        N=N_horizon,
        alpha=1.0,
        beta=1.0,
        gamma=1.0,
        ground_clearance=ground_clearance,
        mass=1.05
    )
    
    # Initial state: pos = [5.0, 5.0, 15.0], vel = [0.0, 0.0, 0.0]
    x_init = [5.0, 5.0, 15.0, 0.0, 0.0, 0.0]
    # Target state: pos = [0.0, 0.0, ground_clearance], vel = [0.0, 0.0, 0.0]
    x_land = [0.0, 0.0, ground_clearance, 0.0, 0.0, 0.0]
    
    print("=" * 60)
    print(f"Solving Guidance MPC at t=0:")
    print(f"  Initial State (6D): {x_init}")
    print(f"  Target Landing (6D): {x_land}")
    print(f"  Horizon Nodes N:    {N_horizon}")
    print("=" * 60)
    
    # 2. Solve NLP
    t_start = time.perf_counter()
    trajectory, t_final = planner.solve(x_init, x_land, t_elapsed=None, t_plan_start=0.0)
    solve_duration = (time.perf_counter() - t_start) * 1000.0
    
    print(f"Solve Successful!")
    print(f"  Computation Time:   {solve_duration:.2f} ms")
    print(f"  Optimal Duration Tf*: {t_final:.3f} s")
    fuel_burned = planner.X_sol[6, 0] - planner.X_sol[6, -1]
    print(f"  Final Position:     [{planner.X_sol[0, -1]:.4f}, {planner.X_sol[1, -1]:.4f}, {planner.X_sol[2, -1]:.4f}] m")
    print(f"  Final Velocity:     [{planner.X_sol[3, -1]:.4f}, {planner.X_sol[4, -1]:.4f}, {planner.X_sol[5, -1]:.4f}] m/s")
    print(f"  Initial Mass:       {planner.X_sol[6, 0]:.4f} kg")
    print(f"  Final Mass:         {planner.X_sol[6, -1]:.4f} kg")
    print(f"  Fuel Consumed:      {fuel_burned:.4f} kg ({fuel_burned / planner.X_sol[6, 0] * 100:.1f}%)")
    print("=" * 60)
    
    # 3. Extract Trajectory Data
    t_nodes = planner.t_grid_sol       # (N+1,)
    X_nodes = planner.X_sol            # (7, N+1)
    U_nodes = planner.U_sol            # (3, N)
    
    px = X_nodes[0, :]
    py = X_nodes[1, :]
    pz = X_nodes[2, :]
    vx = X_nodes[3, :]
    vy = X_nodes[4, :]
    vz = X_nodes[5, :]
    m_nodes = X_nodes[6, :]
    vel_mag = np.linalg.norm(X_nodes[3:6, :], axis=0)
    thrust_mag = np.linalg.norm(U_nodes, axis=0)
    
    # Dense continuous interpolation for smooth line
    t_dense = np.linspace(0.0, t_final, 300)
    dense_samples = np.array([trajectory.sample_state_7d(t) for t in t_dense]).T  # (7, 300)
    
    # 4. Visualization: 3D Trajectory & Mass/Thrust Profiles
    fig = plt.figure(figsize=(16, 7))
    ax = fig.add_subplot(121, projection='3d')
    
    # Draw ground plane (z = 0) and landing altitude plane (z = ground_clearance)
    grid_lim = 6.0
    gx, gy = np.meshgrid(np.linspace(-1, grid_lim, 10), np.linspace(-1, grid_lim, 10))
    ax.plot_surface(gx, gy, np.full_like(gx, ground_clearance), alpha=0.15, color='green')
    ax.plot_surface(gx, gy, np.zeros_like(gx), alpha=0.1, color='gray')
    
    # Smooth continuous curve
    ax.plot(
        dense_samples[0, :],
        dense_samples[1, :],
        dense_samples[2, :],
        color='royalblue',
        linestyle='--',
        linewidth=2,
        label='Spline Trajectory'
    )
    
    # Discrete optimization knot points colored by velocity magnitude
    scatter = ax.scatter(
        px, py, pz,
        c=vel_mag,
        cmap='plasma',
        s=45,
        edgecolor='k',
        alpha=0.9,
        label=f'Nodes (N={N_horizon})'
    )
    
    # Start & Target Points
    ax.scatter([x_init[0]], [x_init[1]], [x_init[2]], color='red', s=120, marker='^', edgecolor='black', label=f'Start ({x_init[0]}, {x_init[1]}, {x_init[2]})')
    ax.scatter([x_land[0]], [x_land[1]], [x_land[2]], color='limegreen', s=140, marker='*', edgecolor='black', label=f'Target ({x_land[0]}, {x_land[1]}, {x_land[2]:.2f})')
    
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1, shrink=0.6)
    cbar.set_label('Velocity Magnitude (m/s)', fontsize=10)
    ax.set_title(f"3D Guidance Trajectory ($T_f^* = {t_final:.2f}$ s)", fontsize=12, fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_xlim([-1.0, 6.0])
    ax.set_ylim([-1.0, 6.0])
    ax.set_zlim([0.0, 16.0])
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.view_init(elev=25, azim=45)
    
    # Subplot 2: Mass Depletion & Thrust Magnitude
    ax2 = fig.add_subplot(122)
    color_m = 'tab:blue'
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('Vehicle Mass (kg)', color=color_m, fontsize=11)
    line_m, = ax2.plot(t_dense, dense_samples[6, :], color=color_m, linewidth=2.5, label='Mass m(t)')
    ax2.tick_params(axis='y', labelcolor=color_m)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.set_ylim([planner.dry_mass * 0.9, planner.X_sol[6, 0] * 1.05])
    ax2.axhline(planner.dry_mass, color='tab:red', linestyle='--', alpha=0.7, label=f'Dry Mass ({planner.dry_mass:.3f} kg)')
    
    # Twin axis for thrust
    ax3 = ax2.twinx()
    color_f = 'tab:orange'
    ax3.set_ylabel('Thrust Magnitude (N)', color=color_f, fontsize=11)
    line_f, = ax3.step(t_nodes[:-1], thrust_mag, where='post', color=color_f, linewidth=2, label='Thrust ||F||')
    ax3.tick_params(axis='y', labelcolor=color_f)
    ax3.set_ylim([0, 22.0])
    
    lines = [line_m, line_f]
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper right', fontsize=10)
    ax2.set_title(rf"Mass Depletion & Thrust Profile ($\Delta m = {fuel_burned:.3f}$ kg)", fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    print("Showing 3D trajectory and mass profile plot...")
    plt.show()


if __name__ == '__main__':
    main()
