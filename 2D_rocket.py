import os
import time
import mujoco as mj
import mujoco.viewer

import rerun as rr
import numpy as np
from scipy.spatial.transform import Rotation as R

from sensor import Accelerometer, Gyroscope, GPS
from multiplicative_ekf import MEKF

# Get absolute path to XML file
XML_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "2D_rocket.xml"))

rr.init("mujoco_telemetry", spawn=True)

m = mujoco.MjModel.from_xml_path(XML_PATH)
d = mujoco.MjData(m)

physics_dt = m.opt.timestep
render_fps = 30
gps_frequency = 10.0
steps_per_render = int((1.0 / render_fps) / physics_dt)
steps_per_gps = int((1.0 / gps_frequency) / physics_dt)

def load_model():
    if not os.path.exists(XML_PATH):
        raise FileNotFoundError(f"XML file not found at: {XML_PATH}")
    
    return m, d

def controller(m, d):
    set_position_servo(0, 100)
    set_torque_servo(1, 1)
    
    d.ctrl[0] = 0
    d.ctrl[1] = 10

def wind_field(model, data):
    pass

def set_position_servo(actuator_id, kp):
    m.actuator_gainprm[actuator_id, 0] = kp
    m.actuator_biasprm[actuator_id, 1] = -kp

def set_torque_servo(actuator_id, flag):
    m.actuator_gainprm[0, 0] = 1

def quat_to_euler(q):
    r = R.from_quat(q)
    euler_angles = r.as_euler('xyz', degrees=True)
    return euler_angles

def main():
    camera_name = "my_cam"
    camera_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    cam.fixedcamid = camera_id

    print("Launching MuJoCo Interactive Viewer...")
    print(f"Loading model from: {XML_PATH}")

    mj.set_mjcb_control(controller)
    #mj.set_mjcb_passive(wind_field)

    accel = Accelerometer(3, 0.025, 0.005)
    gyro = Gyroscope(3, 0.005, 0.001)
    gps = GPS(3, 0.001)

    mj.mj_forward(m, d)
    
    x0, z0, phi0, gimbal0 = d.qpos
    vx0, vz0, vphi0, vgimbal0 = d.qvel

    p0 = np.array([x0, 0, z0]).reshape(-1, 1)
    v0 = np.array([vx0, 0, vz0]).reshape(-1, 1)

    r = R.from_rotvec(np.array([0, -phi0, 0]))
    q0_scipy = r.as_quat() # [x, y, z, w]
    q0 = np.array([q0_scipy[3], q0_scipy[0], q0_scipy[1], q0_scipy[2]]).reshape(-1, 1) # [w, x, y, z]

    # Covariances for MEKF (gyro, accel, gyro_bias, accel_bias)
    gyro_cov = 0.05
    accel_cov = 0.1
    gyro_bias_cov = 0.001
    accel_bias_cov = 0.001

    # GPS measurement covariance (much smaller relative to IMU noise)
    gps_cov = 0.001

    # Initialize P with sensible guesses
    P_initial = np.zeros((15, 15), dtype=float)
    np.fill_diagonal(P_initial[0:3, 0:3], 0.01) # orientation
    np.fill_diagonal(P_initial[3:6, 3:6], 1.0) # velocity
    np.fill_diagonal(P_initial[6:9, 6:9], 1.0) # position
    np.fill_diagonal(P_initial[9:12, 9:12], 0.001) # gyro bias
    np.fill_diagonal(P_initial[12:15, 12:15], 0.001) # accel bias

    mekf = MEKF(q0, v0, p0, gyro_cov, accel_cov, gyro_bias_cov, accel_bias_cov, gps_cov=gps_cov, P_initial=P_initial)

    steps_to_render = steps_per_render
    steps_to_gps = steps_per_gps

    with mujoco.viewer.launch_passive(m, d) as viewer:
        #viewer.cam.type = cam.type
        #viewer.cam.fixedcamid = cam.fixedcamid

        while viewer.is_running():
            step_start = time.time()

            # print("x: ", d.qpos[0])
            # print("z: ", d.qpos[1])
            # print("phi: ", d.qpos[2])

            # sensor sampling
            sensor_acc = accel.sample(np.array([d.qacc[0], 0, d.qacc[1]]), physics_dt)
            sensor_gyro = gyro.sample(d.sensordata[:3], physics_dt)

            # mekf update
            mekf.prediction(physics_dt, sensor_gyro.reshape(3, 1), sensor_acc.reshape(3, 1))
    
            mj.mj_step(m, d)

            viewer.sync()

            if steps_to_gps >= steps_per_gps:
                gps_pos = gps.sample(np.array([d.qpos[0], 0, d.qpos[1]]).reshape(3, 1))
                mekf.correction(physics_dt * steps_to_gps, gps=gps_pos)
                steps_to_gps = 0

            if steps_to_render >= steps_per_render:
                rr.set_time("sim_time", sequence=int(d.time/physics_dt))

                # log clean vs. noisy states
                rr.log("sensors/accelerometer/clean_x", rr.Scalars(round(d.qacc[0], 3)))
                rr.log("sensors/accelerometer/sensor_x", rr.Scalars(sensor_acc[0]))
                rr.log("sensors/accelerometer/clean_z", rr.Scalars(round(d.qacc[1], 3)))
                rr.log("sensors/accelerometer/sensor_z", rr.Scalars(sensor_acc[1]))

                # log mekf states
                rr.log("mekf/mean/x", rr.Scalars(mekf.mean[6]))
                rr.log("mekf/mean/z", rr.Scalars(mekf.mean[8]))
                rr.log("mekf/mean/phi", rr.Scalars(mekf.mean[1]))
                rr.log("mekf/mean/vx", rr.Scalars(mekf.mean[3]))
                rr.log("mekf/mean/vz", rr.Scalars(mekf.mean[5]))

                steps_to_render = 0

            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            steps_to_render += 1
            steps_to_gps += 1
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()