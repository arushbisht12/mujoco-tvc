import time
import mujoco
import mujoco.viewer
import numpy as np

def main():
    
    m = mujoco.MjModel.from_xml_path("mjcf.xml")
    d = mujoco.MjData(m)

    print("Launching MuJoCo Interactive Viewer...")
    print("Press the Spacebar to pause/unpause the physics. Press ESC to exit.")
    
    with mujoco.viewer.launch_passive(m, d) as viewer:
        start_time = time.time()
        
        while viewer.is_running() and time.time() - start_time < 30.0:
            step_start = time.time()
            
            mujoco.mj_step(m, d)
            
            viewer.sync()
            
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()