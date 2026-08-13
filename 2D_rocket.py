import os
import time
import mujoco as mj
import mujoco.viewer

import rerun as rr
import numpy as np
from scipy.spatial.transform import Rotation as R

from sensor import Accelerometer, Gyroscope

# Get absolute path to XML file
XML_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "2D_rocket.xml"))

rr.init("mujoco_telemetry", spawn=True)

m = mujoco.MjModel.from_xml_path(XML_PATH)
d = mujoco.MjData(m)

physics_dt = m.opt.timestep
render_fps = 50
steps_per_render = int((1.0 / render_fps) / physics_dt)

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

    accel = Accelerometer(3, 0.1, 0.05)
    gyro = Gyroscope(3, 0.05, 0.025)

    steps_to_render = 0

    with mujoco.viewer.launch_passive(m, d) as viewer:
        #viewer.cam.type = cam.type
        #viewer.cam.fixedcamid = cam.fixedcamid

        while viewer.is_running():
            step_start = time.time()

            print("x: ", d.qpos[0])
            print("z: ", d.qpos[1])
            print("phi: ", d.qpos[2])
    
            mj.mj_step(m, d)

            viewer.sync()

            if steps_to_render >= steps_per_render:
                rr.set_time("sim_time", sequence=int(d.time/physics_dt))
                
                sensor_acc = accel.sample(d.qacc[:3], physics_dt)

                # Log 2D Time Series: Clean vs. Noisy Sensor Data
                rr.log("sensors/accelerometer/clean_x", rr.Scalars(d.qacc[0]))
                rr.log("sensors/accelerometer/sensor_x", rr.Scalars(sensor_acc[0]))
                rr.log("sensors/accelerometer/clean_z", rr.Scalars(d.qacc[1]))
                rr.log("sensors/accelerometer/sensor_z", rr.Scalars(sensor_acc[1]))
                #rr.log("sensors/accelerometer/noisy_x", rr.Scalar(d.qpos[0]))

                """# Log 3D Transform: Ground Truth
                gt_pos = d.qpos[:3]
                gt_quat_wxyz = d.qpos[3:7]
                rr.log("world/robot/ground_truth", rr.Transform3D(
                    translation=gt_pos,
                    quaternion=[gt_quat_wxyz[1], gt_quat_wxyz[2], gt_quat_wxyz[3], gt_quat_wxyz[0]] # wxyz -> xyzw
                ))

                # Log 3D Transform: EKF Belief ("Ghost")
                rr.log("world/robot/ekf_belief", rr.Transform3D(
                    translation=ekf_pos,
                    quaternion=[ekf_quat_wxyz[1], ekf_quat_wxyz[2], ekf_quat_wxyz[3], ekf_quat_wxyz[0]]
                ))"""

            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            steps_to_render += 1
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()