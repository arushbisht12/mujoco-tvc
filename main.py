import os
import time
import mujoco as mj
import mujoco.viewer

import numpy as np

# Get absolute path to XML file
XML_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "manipulator.xml"))

def load_model():
    if not os.path.exists(XML_PATH):
        raise FileNotFoundError(f"XML file not found at: {XML_PATH}")
    
    return m, d

def main():
    m = mujoco.MjModel.from_xml_path(XML_PATH)
    d = mujoco.MjData(m)

    camera_name = "my_cam"
    camera_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    cam.fixedcamid = camera_id

    print("Launching MuJoCo Interactive Viewer...")
    print(f"Loading model from: {XML_PATH}")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        viewer.cam.type = cam.type
        viewer.cam.fixedcamid = cam.fixedcamid

        N = 1000
        q0_start = 0
        q0_end = 1.57
        q1_start = 0
        q1_end = 1.57
        q0 = np.linspace(q0_start, q0_end, N)
        q1 = np.linspace(q1_start, q1_end, N)
        
        #initialize
        d.qpos[0] = q0_start
        d.qpos[1] = q1_start

        i = 0

        while viewer.is_running():
            step_start = time.time()

            if i < N:
                d.qpos[0] = q0[i]
                d.qpos[1] = q1[i]
                mj.mj_forward(m, d)
                i += 1

            print(d.site_xpos[0])

            
            viewer.sync()

            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()