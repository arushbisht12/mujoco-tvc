import os
import time
import multiprocessing as mp

import mujoco as mj
import mujoco.viewer

import rerun as rr
import rerun.blueprint as rrb
import numpy as np
from scipy.spatial.transform import Rotation as R

from sensor import Accelerometer, Gyroscope, GPS, Magnetometer
from multiplicative_ekf import MEKF
from control import MPCController
from planner import MPCPlanner, GuidanceTrajectory, guidance_worker_loop
from util import quat_to_euler, quat_rotation_matrix

XML_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "2D_rocket.xml"))

rr.init("mujoco_telemetry", spawn=True)

m = mj.MjModel.from_xml_path(XML_PATH)
d = mj.MjData(m)

physics_dt = m.opt.timestep
render_fps = 30
gps_frequency = 10.0
steps_per_render = int((1.0 / render_fps) / physics_dt)
steps_per_gps = int((1.0 / gps_frequency) / physics_dt)

# Propellant Depletion & Variable Mass Settings
ENABLE_FUEL_DEPLETION = True
ISP = 120.0             # Specific impulse in seconds (fuel burn rate: dm = F / (ISP * g0) * dt)
MIN_SLOSH_MASS = 0.005  # Residual dry slosh floor (kg)
MIN_FLUID_MASS = 0.005
G0 = 9.81               # Standard gravity constant (m/s^2)

def load_model():
    if not os.path.exists(XML_PATH):
        raise FileNotFoundError(f"XML file not found at: {XML_PATH}")
    return m, d

mpc = None
planner = None
tracker = None
x_target = None
x_land = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
last_mpc_time = -1.0
last_guidance_time = -1.0
guidance_plan_start_time = 0.0
last_tracker_time = -1.0
t_final_current = 0.0
current_thrust = 0.0
current_gimbal_x = 0.0
current_gimbal_y = 0.0
mekf = None
use_mekf = True
latest_sensor_gyro = None
gyro_accumulator = []  # Accumulate gyro readings between MPC steps for averaging

waypoints = []
current_waypoint_index = 0
waypoint_arrival_time = None
engine_cut = False
landing_evaluated = False

def init_controller(m, d):
    global mpc, x_target, current_thrust, waypoints
    total_mass = float(np.sum(m.body_mass))
    mpc = MPCController(N=20, dt=0.1, alpha=0.24, beta=0.04, gamma=3.12, mass=total_mass)
    # Target state (13D): [px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
    x_target = [2.0, 2.0, 10.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    waypoints = []

    current_thrust = -m.opt.gravity[2] * total_mass # setpoint thrust

def init_trajectory_controller(m, d):
    global mpc, x_target, current_thrust, waypoints, current_waypoint_index, waypoint_arrival_time
    total_mass = float(np.sum(m.body_mass))
    mpc = MPCController(N=15, dt=0.1, alpha=16.68276963291755, beta=0.5843884594990272, gamma=4.485854411546921, mass=total_mass)
    
    waypoints = [
        [2.0, 0.0, 15.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Up and to the right
        [-2.0, 2.0, 10.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # Down and to the left (lateral shift)
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # Land at origin
    ]
    current_waypoint_index = 0
    x_target = [d.qpos[0], d.qpos[1], d.qpos[2], 0.0, 0.0, 0.0, d.qpos[3], d.qpos[4], d.qpos[5], d.qpos[6], 0.0, 0.0, 0.0] # initial pos
    waypoint_arrival_time = None
    
    current_thrust = -m.opt.gravity[2] * total_mass

def initialize_controller1(m, d):
    """
    Initializes the 2-level controller:
    - High-level guidance worker process (free final time optimization)
    - Low-level trajectory-tracking MPCController (6-DOF rigid body MPC)
    """
    global tracker, x_land, last_guidance_time, last_tracker_time
    global current_thrust, current_gimbal_x, current_gimbal_y, t_final_current, engine_cut, landing_evaluated
    global x_target, ground_clearance
    global worker_process, guidance_req_queue, guidance_res_queue, latest_trajectory, guidance_is_busy
    
    # Infer ground clearance from XML (half-length of the rocket's main cylinder)
    rocket_body_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "rocket")
    geom_id = m.body_geomadr[rocket_body_id]
    ground_clearance = float(m.geom_size[geom_id, 1]) + 0.25

    total_mass = float(np.sum(m.body_mass))
    
    guidance_req_queue = mp.Queue(maxsize=1)
    guidance_res_queue = mp.Queue(maxsize=1)
    latest_trajectory = None
    guidance_is_busy = False

    planner_config = {
        "N": 60,
        "alpha": 1.0,
        "beta": 1.0,
        "gamma": 1.0,
        "ground_clearance": ground_clearance,
        "mass": total_mass,
        "dry_mass": 0.20 * total_mass + 0.005,
        "isp": ISP
    }
    worker_process = mp.Process(
        target=guidance_worker_loop,
        args=(guidance_req_queue, guidance_res_queue, planner_config),
        daemon=True
    )
    worker_process.start()
    
    
    tracker = MPCController(
        N=15,
        dt=0.1,
        alpha=16.68276963291755,
        beta=0.5843884594990272,
        gamma=4.485854411546921,
        ground_clearance=ground_clearance,
        mass=total_mass
    )
    
    # Target landing state (6D): [px, py, pz, vx, vy, vz]
    x_land = [0.0, 0.0, ground_clearance, 0.0, 0.0, 0.0]
    
    current_thrust = -m.opt.gravity[2] * total_mass
    current_gimbal_x = 0.0
    current_gimbal_y = 0.0
    
    last_guidance_time = -1.0
    last_tracker_time = -1.0
    t_final_current = 1e4
    engine_cut = False
    landing_evaluated = False
    x_target = [d.qpos[0], d.qpos[1], d.qpos[2], 0.0, 0.0, 0.0, d.qpos[3], d.qpos[4], d.qpos[5], d.qpos[6], 0.0, 0.0, 0.0]

    # Trigger initial guidance plan request
    x_init_6d = [d.qpos[0], d.qpos[1], d.qpos[2], 0.0, 0.0, 0.0]
    guidance_req_queue.put((d.time, x_init_6d, x_land, None, total_mass))
    guidance_is_busy = True
    last_guidance_time = d.time
    print("Initialized 2-Level Controller: High-Level Guidance + Low-Level Tracker MPC. Running in parallel processes.")

def controller1(m, d):
    """
    2-Level Hierarchical Controller:
    - Guidance planner runs asynchronously in worker process (~2 Hz)
    - Tracking MPC runs at 40 Hz sampling continuous interpolated trajectory
    """
    global tracker, x_land, last_guidance_time, last_tracker_time
    global current_thrust, current_gimbal_x, current_gimbal_y, t_final_current, engine_cut, landing_evaluated
    global mekf, latest_sensor_gyro, use_mekf, gyro_accumulator, x_target, ground_clearance
    global guidance_req_queue, guidance_res_queue, latest_trajectory, guidance_is_busy

    # Extract state feedback from MEKF or ground truth
    current_total_mass = float(np.sum(m.body_mass))
    if use_mekf and mekf is not None:
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
            
        x_current_6d = [px, py, pz, vx, vy, vz]
        x_current_13d = [px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
    else:
        q = d.qpos[3:7]
        qw, qx, qy, qz = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        R_b2w = quat_rotation_matrix([qw, qx, qy, qz])
        omega_body = R_b2w.T @ np.array([d.qvel[3], d.qvel[4], d.qvel[5]])
        px, py, pz = float(d.qpos[0]), float(d.qpos[1]), float(d.qpos[2])
        vx, vy, vz = float(d.qvel[0]), float(d.qvel[1]), float(d.qvel[2])
        wx, wy, wz = float(omega_body[0]), float(omega_body[1]), float(omega_body[2])
        
        x_current_6d = [px, py, pz, vx, vy, vz]
        x_current_13d = [px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]

    # 1. Check for incoming guidance solutions from worker (non-blocking)
    if guidance_res_queue is not None and not guidance_res_queue.empty():
        try:
            res = guidance_res_queue.get_nowait()
            latest_trajectory = res["trajectory"]
            t_final_current = res["t_final"]
            guidance_is_busy = False
            lag_ms = (d.time - res["t_req"]) * 1000.0
            print(f"[Guidance] Received plan at t={d.time:.2f}s | Tf*={t_final_current:.2f}s | Solve lag={lag_ms:.1f}ms")
        except mp.queues.Empty:
            pass

    # 2. Trigger next open-loop guidance update at 2 Hz
    if (d.time - last_guidance_time >= 0.5) and not guidance_is_busy and t_final_current > 3.0:
        t_elapsed = d.time - last_guidance_time
        try:
            # Open-loop step: send x_init=None so planner samples state and mass from previous trajectory
            guidance_req_queue.put_nowait((d.time, None, x_land, t_elapsed))
            guidance_is_busy = True
            last_guidance_time = d.time
        except mp.queues.Full:
            pass

    # 3. Low-Level Tracking MPC update (40 Hz)
    if d.time - last_tracker_time >= 0.025:
        gyro_accumulator.clear()  # reset gyro accumulator for next period
        
        # Sample the continuous interpolated trajectory over tracking preview horizon
        if latest_trajectory is not None:
            traj_ref_13d = latest_trajectory.sample_13d(
                d.time,
                N_low=tracker.N,
                dt_low=tracker.dt
            )
        else:
            # Upright hover default while waiting for initial plan
            traj_ref_13d = np.zeros((13, tracker.N + 1))
            traj_ref_13d[:3, :] = np.array([x_current_6d[0], x_current_6d[1], x_current_6d[2]])[:, None]
            traj_ref_13d[6, :] = 1.0  # qw = 1.0
            
        # Update current reference setpoint for visual telemetry
        x_target = traj_ref_13d[:3, 0].tolist()
        
        t_track_start = time.perf_counter()
        current_thrust, current_gimbal_x, current_gimbal_y = tracker.solve(
            x_current_13d, traj_ref_13d, current_mass=current_total_mass
        )
        t_track_loop = time.perf_counter() - t_track_start
        last_tracker_time = d.time

    # 4. Touchdown & Landing Evaluation
    if pz < ground_clearance + 0.05:
        vel_mag = np.linalg.norm([vx, vy, vz])
        if not engine_cut:
            if vel_mag < 0.2: 
                print(f"Landing successful! Velocity: {vel_mag:.2f} m/s. Cutting engine.")
                print(f"Final Position: ({px:.2f}, {py:.2f}, {pz:.2f})")
                engine_cut = True
            else:
                print(f"Landing alert: Touchdown velocity: {vel_mag:.2f} m/s.")

    # 5. Actuation
    if engine_cut:
        d.ctrl[0] = 0.0
        d.ctrl[1] = 0.0
        d.ctrl[2] = 0.0
    else:
        d.ctrl[0] = current_gimbal_x
        d.ctrl[1] = current_gimbal_y
        d.ctrl[2] = current_thrust

def controller(m, d):
    global last_mpc_time, current_thrust, current_gimbal_x, current_gimbal_y, mekf, latest_sensor_gyro, use_mekf, gyro_accumulator
    global x_target, current_waypoint_index, waypoint_arrival_time, waypoints
    global engine_cut, landing_evaluated

    # 10 Hz MPC update
    if d.time - last_mpc_time >= 0.1:
        
        # dumb high level planner (interpolates between waypoints)
        if len(waypoints) > 0 and x_target is not None:
            target_waypoint = np.array(waypoints[current_waypoint_index])
            current_x_target = np.array(x_target)
            
            diff = target_waypoint[:3] - current_x_target[:3]
            dist_to_waypoint = np.linalg.norm(diff)
            
            # max step size per MPC update (0.1s): 0.3 meters -> 3.0 m/s velocity
            step_size = 0.3
            if dist_to_waypoint > step_size:
                current_x_target[:3] += (diff / dist_to_waypoint) * step_size
            else:
                current_x_target[:3] = target_waypoint[:3]
                
            x_target = current_x_target.tolist()
            
            # check if reached waypoint
            px, py, pz = d.qpos[0], d.qpos[1], d.qpos[2]
            dist_physical = np.linalg.norm(np.array([px, py, pz]) - target_waypoint[:3])
            
            if dist_physical < 0.6 and dist_to_waypoint < step_size:
                if current_waypoint_index == len(waypoints) - 1: # trajectory end?
                    if not landing_evaluated:
                        landing_evaluated = True
                        vel_mag = np.linalg.norm([d.qvel[0], d.qvel[1], d.qvel[2]])
                        if vel_mag < 2.5: # max safe velocity threshold
                            print(f"Landing successful! Velocity: {vel_mag:.2f} m/s. Cutting engine.")
                            engine_cut = True
                        else:
                            print(f"Landing failed. Velocity too high: {vel_mag:.2f} m/s.")
                
                if waypoint_arrival_time is None:
                    waypoint_arrival_time = d.time
                elif d.time - waypoint_arrival_time > 2.0: # hover for 2 sec
                    if current_waypoint_index < len(waypoints) - 1:
                        current_waypoint_index += 1
                        waypoint_arrival_time = None

        # state feedback
        if use_mekf and mekf is not None:
            px, py, pz = mekf.pos[0], mekf.pos[1], mekf.pos[2]
            vx, vy, vz = mekf.vel[0], mekf.vel[1], mekf.vel[2]
            qw, qx, qy, qz = mekf.q[0], mekf.q[1], mekf.q[2], mekf.q[3]
            
            # average gyro reading over X 
            if len(gyro_accumulator) > 0:
                gyro_avg = np.mean(gyro_accumulator, axis=0)
                wx = gyro_avg[0] - mekf.gyr_b[0] # mekf estimates bias as well
                wy = gyro_avg[1] - mekf.gyr_b[1]
                wz = gyro_avg[2] - mekf.gyr_b[2]
            else:
                # attitude fallback: d.qvel[3:6] is WORLD frame omega, rotate to body frame
                R_b2w = quat_rotation_matrix([qw, qx, qy, qz])
                omega_body = R_b2w.T @ np.array([d.qvel[3], d.qvel[4], d.qvel[5]])
                wx, wy, wz = omega_body[0], omega_body[1], omega_body[2]
            
            gyro_accumulator.clear()  # reset for next MPC period
            
            x_current = [px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
        else:
            # ground truth state
            q = d.qpos[3:7]
            qw, qx, qy, qz = q[0], q[1], q[2], q[3]
            # d.qvel[3:6] is WORLD frame omega for free joints; rotate to body frame TODO could just directly use gyro
            R_b2w = quat_rotation_matrix(q)
            omega_body = R_b2w.T @ np.array([d.qvel[3], d.qvel[4], d.qvel[5]])
            x_current = [
                d.qpos[0], d.qpos[1], d.qpos[2],
                d.qvel[0], d.qvel[1], d.qvel[2],
                qw, qx, qy, qz,
                omega_body[0], omega_body[1], omega_body[2]
            ]
        t_mpc_start = time.perf_counter()
        current_thrust, current_gimbal_x, current_gimbal_y = mpc.solve(
            x_current, x_target, current_mass=float(np.sum(m.body_mass))
        )
        t_mpc_loop = time.perf_counter() - t_mpc_start
        print(f"MPC loop time: {t_mpc_loop * 1000:.2f} ms | Max Freq: {1/(t_mpc_loop):.2f} Hz")
        last_mpc_time = d.time
        
    if engine_cut:
        d.ctrl[0] = 0.0
        d.ctrl[1] = 0.0
        d.ctrl[2] = 0.0
    else:
        d.ctrl[0] = current_gimbal_x
        d.ctrl[1] = current_gimbal_y
        d.ctrl[2] = current_thrust


def main():
    global mekf, latest_sensor_gyro

    camera_name = "my_cam"
    camera_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_CAMERA, camera_name)

    cam = mj.MjvCamera()
    cam.type = mj.mjtCamera.mjCAMERA_FIXED
    cam.fixedcamid = camera_id

    print("Launching MuJoCo Interactive Viewer...")
    print(f"Loading model from: {XML_PATH}")

    accel = Accelerometer(3, 0.005, 0.001)
    gyro = Gyroscope(3, 0.005, 0.002)
    gps = GPS(3, 0.001)
    mag = Magnetometer(3, 0.01, 0.0)

    mj.mj_forward(m, d)
    
    p0 = np.array(d.qpos[0:3]).reshape(3, 1)
    v0 = np.array(d.qvel[0:3]).reshape(3, 1)
    q0 = np.array(d.qpos[3:7]).reshape(4, 1) # [w, x, y, z]

    # Covariances for MEKF (gyro, accel, gyro_bias, accel_bias)
    gyro_cov = 0.05
    accel_cov = 0.1
    gyro_bias_cov = 0.001
    accel_bias_cov = 0.001

    # GPS measurement covariance
    gps_cov = 0.001

    # Initialize covariance matrix P with guesses
    P_initial = np.zeros((15, 15), dtype=float)
    np.fill_diagonal(P_initial[0:3, 0:3], 0.01)   # orientation
    np.fill_diagonal(P_initial[3:6, 3:6], 1.0)    # velocity
    np.fill_diagonal(P_initial[6:9, 6:9], 1.0)    # position
    np.fill_diagonal(P_initial[9:12, 9:12], 0.001) # gyro bias
    np.fill_diagonal(P_initial[12:15, 12:15], 0.001) # accel bias

    # Align MEKF north_vector with MuJoCo's magnetic field direction
    # MuJoCo default: model.opt.magnetic = [0, -0.5, 0] (world frame)
    # MEKF north_vector must match this direction (normalized)
    mag_world = np.array(m.opt.magnetic)
    mag_norm = np.linalg.norm(mag_world)

    mekf = MEKF(q0, v0, p0, gyro_cov, accel_cov, gyro_bias_cov, accel_bias_cov, gps_cov=gps_cov, P_initial=P_initial)

    # Override MEKF's north_vector to match MuJoCo's magnetic field direction
    if mag_norm > 1e-6:
        mekf.north_vector = (mag_world / mag_norm).reshape(3, 1)
        print(f"MEKF north_vector aligned to MuJoCo magnetic field: {mekf.north_vector.flatten()}")

    initialize_controller1(m, d)
    mj.set_mjcb_control(controller1)

    # Slosh and vehicle body setup for propellant depletion
    rocket_body_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "rocket")
    slosh_body_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "slosh_mass_xy")
    slosh_dummy_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "slosh_dummy_x")
    gimbal_pitch_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "gimbal_pitch")
    gimbal_yaw_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "gimbal_yaw")

    initial_total_mass = float(np.sum(m.body_mass))
    initial_slosh_mass = float(m.body_mass[slosh_body_id])
    initial_fluid_mass = 0.8 * initial_total_mass
    initial_solid_mass = 0.2 * initial_total_mass
    current_fluid_mass = initial_fluid_mass
    current_slosh_mass = initial_slosh_mass
    initial_tank_level = 0.2
    tank_radius = 0.08
    # Precomputed sphere inertia factor: I = 2/5 * m * r^2 (sphere geom radius is 0.08 m)
    sphere_geom_id = m.body_geomadr[slosh_body_id]
    sphere_radius = float(m.geom_size[sphere_geom_id, 0]) if sphere_geom_id >= 0 else 0.08
    sphere_inertia_factor = 0.4 * (sphere_radius ** 2)

    other_bodies_mass = float(
        m.body_mass[slosh_dummy_id] + m.body_mass[gimbal_pitch_id] + m.body_mass[gimbal_yaw_id]
    )

    # Rerun blueprint (3-column layout including mass telemetry)
    common_x = rrb.TimeAxis(
        view_range=rr.TimeRange(
            start=rrb.TimeRangeBoundary.absolute(seq=0),
            end=rrb.TimeRangeBoundary.absolute(seq=15000)
        )
    )

    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.TimeSeriesView(name="Position X", origin="position/x", axis_x=common_x, axis_y=rrb.ScalarAxis(range=(-5.0, 5.0))),
                rrb.TimeSeriesView(name="Position Y", origin="position/y", axis_x=common_x, axis_y=rrb.ScalarAxis(range=(-5.0, 5.0))),
                rrb.TimeSeriesView(name="Position Z", origin="position/z", axis_x=common_x, axis_y=rrb.ScalarAxis(range=(0.0, 20.0)))
            ),
            rrb.Vertical(
                rrb.TimeSeriesView(name="Attitude", origin="attitude", axis_x=common_x, axis_y=rrb.ScalarAxis(range=(-0.5, 0.5))),
                rrb.TimeSeriesView(name="Velocity", origin="velocity", axis_x=common_x, axis_y=rrb.ScalarAxis(range=(-10.0, 10.0))),
                rrb.TimeSeriesView(name="Controls", origin="control", axis_x=common_x)
            ),
            rrb.Vertical(
                rrb.TimeSeriesView(name="Total Mass (kg)", origin="mass/total", axis_x=common_x, axis_y=rrb.ScalarAxis(range=(0.6, 1.1))),
                rrb.TimeSeriesView(name="Fluid Propellant (kg)", origin="mass/fluid_propellant", axis_x=common_x, axis_y=rrb.ScalarAxis(range=(0.0, 1.0))),
                rrb.TimeSeriesView(name="Slosh Propellant (kg)", origin="mass/slosh_propellant", axis_x=common_x, axis_y=rrb.ScalarAxis(range=(0.0, 0.3))),
                rrb.TimeSeriesView(name="Optimal Time-To-Go", origin="guidance/t_final", axis_x=common_x)
            )
        )
    )
    rr.send_blueprint(blueprint)

    steps_to_render = steps_per_render
    steps_to_gps = steps_per_gps

    try:
        with mj.viewer.launch_passive(m, d, show_left_ui=True, show_right_ui=True) as viewer:
            # startup delay
            wait_time = 5.0
            wall_start_time = time.time()
            print(f"Waiting {wait_time} seconds for you to arrange windows...")
            
            while viewer.is_running():
                # start up delay
                step_start = time.time()
                if time.time() - wall_start_time < wait_time:
                    viewer.sync()
                    time.sleep(1/60.0)
                    continue

                # sample IMU (high freq sensors)
                sensor_acc = accel.sample(d.sensordata[:3], physics_dt)
                sensor_gyro = gyro.sample(d.sensordata[3:6], physics_dt)
                sensor_mag = mag.sample(d.sensordata[6:9])
                latest_sensor_gyro = sensor_gyro
                gyro_accumulator.append(sensor_gyro.copy())  # Accumulate for MPC-period averaging

                mekf.prediction(physics_dt, sensor_gyro.reshape(3, 1), sensor_acc.reshape(3, 1))
        
                mj.mj_step(m, d)

                # Propellant depletion: burn fluid mass according to physical engine thrust
                if ENABLE_FUEL_DEPLETION and current_fluid_mass > MIN_FLUID_MASS:
                    actual_thrust = max(0.0, float(d.actuator_force[2]))
                    dm = (actual_thrust / (ISP * G0)) * physics_dt
                    current_fluid_mass = max(MIN_FLUID_MASS, current_fluid_mass - dm)
                    current_total_mass = initial_solid_mass + current_fluid_mass

                    h = max(1e-4, (current_fluid_mass / initial_fluid_mass) * initial_tank_level)
                    current_slosh_mass = max(
                        MIN_SLOSH_MASS,
                        (0.4545 * (tank_radius / h) * np.tanh(1.8412 * h / tank_radius)) * current_fluid_mass
                    )
                    current_rigid_mass = current_total_mass - current_slosh_mass
                    current_rocket_mass = max(0.01, current_rigid_mass - other_bodies_mass)

                    # 1. Update slosh mass sphere body
                    m.body_mass[slosh_body_id] = current_slosh_mass
                    m.body_inertia[slosh_body_id, :] = sphere_inertia_factor * current_slosh_mass

                    # 2. Update rocket cylinder body (structure + static fluid)
                    m.body_mass[rocket_body_id] = current_rocket_mass
                    m.body_inertia[rocket_body_id, 0] = (1.03 / 12.0) * current_rocket_mass
                    m.body_inertia[rocket_body_id, 1] = (1.03 / 12.0) * current_rocket_mass
                    m.body_inertia[rocket_body_id, 2] = 0.005 * current_rocket_mass

                    # 3. Update kinematic subtree masses so MuJoCo's center-of-mass kinematics stay exact
                    m.body_subtreemass[slosh_body_id] = current_slosh_mass
                    m.body_subtreemass[slosh_dummy_id] = m.body_mass[slosh_dummy_id] + current_slosh_mass
                    m.body_subtreemass[1] = np.sum(m.body_mass[1:])
                    m.body_subtreemass[0] = m.body_subtreemass[1]

                viewer.sync()

                if steps_to_gps >= steps_per_gps:
                    gps_pos = gps.sample(np.array(d.qpos[0:3]).reshape(3, 1))
                    mekf.correction(physics_dt * steps_to_gps, acc=sensor_acc.reshape(3, 1), mag=sensor_mag.reshape(3, 1), gps=gps_pos)
                    steps_to_gps = 0

                    if engine_cut:
                        print(f"Final Position: ({mekf.pos})")

                if steps_to_render >= steps_per_render:
                    rr.set_time("sim_time", sequence=int(d.time/physics_dt))

                    # Log clean vs estimated positions
                    rr.log("position/x/true", rr.Scalars(mekf.pos[0]))
                    rr.log("position/y/true", rr.Scalars(mekf.pos[1]))
                    rr.log("position/z/true", rr.Scalars(mekf.pos[2]))
                    
                    # Log target positions
                    if x_target is not None:
                        rr.log("position/x/target", rr.Scalars(x_target[0]))
                        rr.log("position/y/target", rr.Scalars(x_target[1]))
                        rr.log("position/z/target", rr.Scalars(x_target[2]))

                    # Log clean vs estimated velocities
                    rr.log("velocity/mekf_vx", rr.Scalars(mekf.vel[0]))
                    rr.log("velocity/mekf_vy", rr.Scalars(mekf.vel[1]))
                    rr.log("velocity/mekf_vz", rr.Scalars(mekf.vel[2]))

                    # Convert to Euler angles only for telemetry logging / visualization
                    euler_mekf = quat_to_euler(mekf.q)
                    rr.log("attitude/roll", rr.Scalars(euler_mekf[0]))
                    rr.log("attitude/pitch", rr.Scalars(euler_mekf[1]))
                    rr.log("attitude/yaw", rr.Scalars(euler_mekf[2]))

                    # Log controls
                    rr.log("control/thrust", rr.Scalars(current_thrust))
                    rr.log("control/gimbal_x", rr.Scalars(current_gimbal_x))
                    rr.log("control/gimbal_y", rr.Scalars(current_gimbal_y))

                    # Log guidance optimal time-to-go
                    rr.log("guidance/t_final", rr.Scalars(t_final_current))

                    # Log mass telemetry
                    total_mass = float(np.sum(m.body_mass))
                    fuel_pct = ((current_fluid_mass - MIN_FLUID_MASS) / max(1e-6, (initial_fluid_mass - MIN_FLUID_MASS))) * 100.0
                    rr.log("mass/total", rr.Scalars(total_mass))
                    rr.log("mass/fluid_propellant", rr.Scalars(current_fluid_mass))
                    rr.log("mass/slosh_propellant", rr.Scalars(current_slosh_mass))
                    rr.log("mass/rigid_cylinder", rr.Scalars(m.body_mass[rocket_body_id]))
                    rr.log("mass/fuel_percent", rr.Scalars(fuel_pct))

                    steps_to_render = 0

                time_until_next_step = m.opt.timestep - (time.time() - step_start)
                steps_to_render += 1
                steps_to_gps += 1
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

    except KeyboardInterrupt:
        print("Simulation closed by user.")
    finally:
        mj.set_mjcb_control(None)
        if guidance_req_queue is not None:
            try:
                guidance_req_queue.put_nowait(None)
            except Exception:   
                pass
        if worker_process is not None and worker_process.is_alive():
            worker_process.terminate()
        try:
            rec = rr.get_global_data_recording()
            if rec is not None:
                rec.flush(timeout_sec=0.5)
            rr.disconnect()
        except Exception:
            pass

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()