import os
import time
import mujoco as mj
import mujoco.viewer

import rerun as rr
import numpy as np
from scipy.spatial.transform import Rotation as R

from sensor import Accelerometer, Gyroscope, GPS
from multiplicative_ekf import MEKF
from control import MPCController
from util import quat_to_euler

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
use_mekf = False
latest_sensor_gyro = None

def init_controller(m, d):
    global mpc, x_target, current_thrust
    mpc = MPCController(N=20, dt=0.1, alpha=0.24, beta=0.04, gamma=3.12)
    # Target state (13D): [px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
    x_target = [2.0, 2.0, 10.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # Hover thrust guess (mass ~ 1.05 kg)
    current_thrust = -m.opt.gravity[2] * 1.05

def controller(m, d):
    global last_mpc_time, current_thrust, current_gimbal_x, current_gimbal_y, mekf, latest_sensor_gyro, use_mekf
    
    # 10 Hz MPC update
    if d.time - last_mpc_time >= 0.1:
        if use_mekf and mekf is not None:
            px, py, pz = mekf.pos[0], mekf.pos[1], mekf.pos[2]
            vx, vy, vz = mekf.vel[0], mekf.vel[1], mekf.vel[2]
            qw, qx, qy, qz = mekf.q[0], mekf.q[1], mekf.q[2], mekf.q[3]
            
            if latest_sensor_gyro is not None:
                wx = latest_sensor_gyro[0] - mekf.gyr_b[0]
                wy = latest_sensor_gyro[1] - mekf.gyr_b[1]
                wz = latest_sensor_gyro[2] - mekf.gyr_b[2]
            else:
                wx, wy, wz = d.qvel[3], d.qvel[4], d.qvel[5]
            
            x_current = [px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
        else:
            # Fallback to ground truth if MEKF not yet ready
            qw, qx, qy, qz = d.qpos[3], d.qpos[4], d.qpos[5], d.qpos[6]
            x_current = [
                d.qpos[0], d.qpos[1], d.qpos[2],
                d.qvel[0], d.qvel[1], d.qvel[2],
                qw, qx, qy, qz,
                d.qvel[3], d.qvel[4], d.qvel[5]
            ]
        
        current_thrust, current_gimbal_x, current_gimbal_y = mpc.solve(x_current, x_target)
        last_mpc_time = d.time
        
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

    # Initialize P with sensible guesses
    P_initial = np.zeros((15, 15), dtype=float)
    np.fill_diagonal(P_initial[0:3, 0:3], 0.01)   # orientation
    np.fill_diagonal(P_initial[3:6, 3:6], 1.0)    # velocity
    np.fill_diagonal(P_initial[6:9, 6:9], 1.0)    # position
    np.fill_diagonal(P_initial[9:12, 9:12], 0.001) # gyro bias
    np.fill_diagonal(P_initial[12:15, 12:15], 0.001) # accel bias

    mekf = MEKF(q0, v0, p0, gyro_cov, accel_cov, gyro_bias_cov, accel_bias_cov, gps_cov=gps_cov, P_initial=P_initial)

    init_controller(m, d)
    mj.set_mjcb_control(controller)

    steps_to_render = steps_per_render
    steps_to_gps = steps_per_gps

    with mj.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # Sensor sampling
            sensor_acc = accel.sample(d.sensordata[:3], physics_dt)
            sensor_gyro = gyro.sample(d.sensordata[3:6], physics_dt)
            latest_sensor_gyro = sensor_gyro

            # MEKF prediction
            mekf.prediction(physics_dt, sensor_gyro.reshape(3, 1), sensor_acc.reshape(3, 1))
    
            mj.mj_step(m, d)

            viewer.sync()

            if steps_to_gps >= steps_per_gps:
                gps_pos = gps.sample(np.array(d.qpos[0:3]).reshape(3, 1))
                mekf.correction(physics_dt * steps_to_gps, acc=sensor_acc.reshape(3, 1), mag=None, gps=gps_pos)
                steps_to_gps = 0

            if steps_to_render >= steps_per_render:
                rr.set_time("sim_time", sequence=int(d.time/physics_dt))

                # Log clean vs estimated positions
                rr.log("mekf/mean/x", rr.Scalars(mekf.pos[0]))
                rr.log("mekf/mean/y", rr.Scalars(mekf.pos[1]))
                rr.log("mekf/mean/z", rr.Scalars(mekf.pos[2]))

                # Log clean vs estimated velocities
                rr.log("velocity/mekf_vx", rr.Scalars(mekf.vel[0]))
                rr.log("velocity/mekf_vy", rr.Scalars(mekf.vel[1]))
                rr.log("velocity/mekf_vz", rr.Scalars(mekf.vel[2]))

                # Convert to Euler angles only for telemetry logging / visualization
                euler_mekf = quat_to_euler(mekf.q)
                rr.log("mekf/mean/roll", rr.Scalars(euler_mekf[0]))
                rr.log("mekf/mean/pitch", rr.Scalars(euler_mekf[1]))
                rr.log("mekf/mean/yaw", rr.Scalars(euler_mekf[2]))

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