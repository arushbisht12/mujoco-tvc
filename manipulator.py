import os
import time
import mujoco as mj
import mujoco.viewer

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Get absolute path to XML file
XML_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "manipulator.xml"))

def load_model():
    if not os.path.exists(XML_PATH):
        raise FileNotFoundError(f"XML file not found at: {XML_PATH}")
    
    return m, d

def controller(m, d):
    pass
    
def main():
    m = mujoco.MjModel.from_xml_path(XML_PATH)
    d = mujoco.MjData(m)

    camera_name = "my_cam"
    camera_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    cam.fixedcamid = camera_id

    #mj.set_mjfc_controller(m, d)

    print("Launching MuJoCo Interactive Viewer...")
    print(f"Loading model from: {XML_PATH}")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        viewer.cam.type = cam.type
        viewer.cam.fixedcamid = cam.fixedcamid

        N = 2000
        xc = 0.75
        yc = 1
        r = 0.25

        thetas = np.linspace(0, np.pi*4,N)
        xs = r*np.cos(thetas) + xc
        ys = r*np.sin(thetas) + yc

        x_all = []
        y_all = []

        # initial
        d.qpos[0] = 0
        d.qpos[1] = 1.57
        
        mj.mj_forward(m, d)

        i = 0

        while viewer.is_running():
            step_start = time.time()

            x, y = d.site_xpos[0][:2]
            x_all.append(x)
            y_all.append(y)
            phi1, phi2 = d.qpos[:2]
            if i < N:
                # get x, y desired
                xd = xs[i]
                yd = ys[i]
            # calculate error
            errors = np.array([xd - x, yd - y])

            # calculate jacobian
            jacp = np.zeros((3,2))
            mj.mj_jac(m, d, jacp, None, d.site_xpos[0], 2)
            J = jacp[[0,1],:] # to only get x and y, no z
            # invert jacobian, solve for dq
            dq = np.linalg.solve(J, errors)
            """Jinv = np.linalg.inv(J)
            dq = Jinv @ errors"""

            phi1 += dq[0]
            phi2 += dq[1]
            d.qpos[0] = phi1  # phi1_dot
            d.qpos[1] = phi2 # phi2_dot

            i += 1

            if i>=N:
                plt.figure()
                plt.plot(x_all, y_all, 'bx')
                plt.plot(xs, ys, 'r-.')
                plt.xlabel("x")
                plt.ylabel("y")
                plt.gca().set_aspect('equal')
                plt.savefig("trajectory.png")
                print("Saved plot to trajectory.png")
                plt.close()
                break

            mj.mj_forward(m, d)
            
            viewer.sync()

            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()