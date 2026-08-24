import os
import time
import mujoco as mj
import mujoco.viewer

import rerun as rr
import rerun.blueprint as rrb
import numpy as np
from scipy.spatial.transform import Rotation as R

from sensor import Accelerometer, Gyroscope, GPS, Magnetometer
from multiplicative_ekf import MEKF
from control import MPCController
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

def load_model():
    if not os.path.exists(XML_PATH):
        raise FileNotFoundError(f"XML file not found at: {XML_PATH}")
    return m, d

mpc = None
x_target = None
last_mpc_time = -1.0
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
    mpc = MPCController(N=20, dt=0.1, alpha=0.24, beta=0.04, gamma=3.12)
    # Target state (13D): [px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
    x_target = [2.0, 2.0, 10.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    waypoints = []

    current_thrust = -m.opt.gravity[2] * 1.05 # setpoint thrust

def init_trajectory_controller(m, d):
    global mpc, x_target, current_thrust, waypoints, current_waypoint_index, waypoint_arrival_time
    mpc = MPCController(N=20, dt=0.1, alpha=0.24, beta=0.04, gamma=3.12)
    
    waypoints = [
        [2.0, 0.0, 15.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Up and to the right
        [-2.0, 2.0, 10.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # Down and to the left (lateral shift)
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # Land at origin
    ]
    current_waypoint_index = 0
    x_target = [d.qpos[0], d.qpos[1], d.qpos[2], 0.0, 0.0, 0.0, d.qpos[3], d.qpos[4], d.qpos[5], d.qpos[6], 0.0, 0.0, 0.0] # initial pos
    waypoint_arrival_time = None
    
    current_thrust = -m.opt.gravity[2] * 1.05

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
        
        current_thrust, current_gimbal_x, current_gimbal_y = mpc.solve(x_current, x_target)
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

    init_trajectory_controller(m, d)
    mj.set_mjcb_control(controller)

    # Rerun blueprint (2x3 grid)
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
            )
        )
    )
    rr.send_blueprint(blueprint)

    steps_to_render = steps_per_render
    steps_to_gps = steps_per_gps

    with mj.viewer.launch_passive(m, d, show_left_ui=False, show_right_ui=False) as viewer:
        # startup delay
        wait_time = 35.0
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

            viewer.sync()

            if steps_to_gps >= steps_per_gps:
                gps_pos = gps.sample(np.array(d.qpos[0:3]).reshape(3, 1))
                mekf.correction(physics_dt * steps_to_gps, acc=sensor_acc.reshape(3, 1), mag=sensor_mag.reshape(3, 1), gps=gps_pos)
                steps_to_gps = 0

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

                steps_to_render = 0

            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            steps_to_render += 1
            steps_to_gps += 1
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()