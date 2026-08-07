import os
import time
import mujoco as mj
import mujoco.viewer

import numpy as np
from scipy.spatial.transform import Rotation as R

# Get absolute path to XML file
XML_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "2D_rocket.xml"))

m = mujoco.MjModel.from_xml_path(XML_PATH)
d = mujoco.MjData(m)

def load_model():
    if not os.path.exists(XML_PATH):
        raise FileNotFoundError(f"XML file not found at: {XML_PATH}")
    
    return m, d

def controller(m, d):
    set_position_servo(0, 100)
    set_torque_servo(1, 1)
    
    d.ctrl[0] = 5
    d.ctrl[1] = 0.1

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

            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()