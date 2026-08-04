import os
import time
import mujoco as mj
import mujoco.viewer

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Get absolute path to XML file
XML_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "ball.xml"))

def init_controller(m, d):
    pass

def controller(m, d): 
    pass


m = mujoco.MjModel.from_xml_path(XML_PATH)
d = mujoco.MjData(m)

init_controller(m, d)
mj.set_mjcb_control(controller)
    
def main():

    print("Launching MuJoCo Interactive Viewer...")
    print(f"Loading model from: {XML_PATH}")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = m.body('ball').id
        viewer.cam.distance = 3.0     # Distance from object
        viewer.cam.elevation = -30    # Vertical angle
        viewer.cam.azimuth = 90
        d.qvel[0] = 1

        while viewer.is_running():
            step_start = time.time()

            mj.mj_step(m, d)
            
            viewer.sync()

            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()