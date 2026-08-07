import os
import time
import mujoco as mj
import mujoco.viewer

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Get absolute path to XML file
XML_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "pendulum.xml"))

FSM_SWINGUP = 0
FSM_HOLD = 1

def init_controller(m, d):
    global FSM
    FSM = FSM_SWINGUP

def controller(m, d):
    
    # spring-like
    """set_position_servo(1, 100)
    d.ctrl[1] = np.pi # 1 because corresponds to actuator 1"""

    # speed control
    """set_velocity_servo(2, 100)
    d.ctrl[2] = 0.5"""

    # pos control
    """set_position_servo(1, 100)
    set_velocity_servo(2, 10)
    d.ctrl[1] = np.pi"""

    # torque
    #set_torque_servo(0, 1)
    #d.ctrl[0] = -10*(d.qpos[0] - np.pi)
    #d.ctrl[0] = -100*(d.qvel[0] - 0.5)
    #d.ctrl[0] =  -100*(d.qpos[0] - np.pi) -10*(d.qvel[0]) # PD control

    global FSM

    if d.qpos[0]>=2.5 and FSM==FSM_SWINGUP:
        FSM = FSM_HOLD

    if FSM==FSM_SWINGUP:
        set_velocity_servo(2, 100)
        d.ctrl[2] = 0.5

    if FSM==FSM_HOLD:
        set_position_servo(1, 100)
        set_velocity_servo(2, 10)
        d.ctrl[1] = np.pi

        


def set_torque_servo(actuator_id, flag):
    m.actuator_gainprm[0, 0] = 1

def set_position_servo(actuator_id, kp):
    m.actuator_gainprm[actuator_id, 0] = kp
    m.actuator_biasprm[actuator_id, 1] = -kp

def set_velocity_servo(actuator_id, kv):
    m.actuator_gainprm[actuator_id, 0] = kv
    m.actuator_biasprm[actuator_id, 2] = -kv

m = mujoco.MjModel.from_xml_path(XML_PATH)
d = mujoco.MjData(m)

init_controller(m, d)
mj.set_mjcb_control(controller)
    
def main():

    print("Launching MuJoCo Interactive Viewer...")
    print(f"Loading model from: {XML_PATH}")

    d.qpos[0] = 0

    with mujoco.viewer.launch_passive(m, d) as viewer:

        while viewer.is_running():
            step_start = time.time()

            

            mj.mj_step(m, d)
            
            viewer.sync()

            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()