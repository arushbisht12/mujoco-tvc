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

q0_init = -3 * np.pi/2
q1_init = 0
q0_end = -np.pi/2
q1_end = np.pi/2

t_init = 1
t_end = 2

t = []
qact0 = []
qref0 = []
qact1 = []
qref1 = []

def init_controller(m, d):
    global a_jnt0, a_jnt1

    a_jnt0 = generate_trajectory(t_init, t_end, q0_init, q0_end)
    a_jnt1 = generate_trajectory(t_init, t_end, q1_init, q1_end)

def controller(m, d):
    global a_jnt0, a_jnt1

    time = d.time
    if time > t_end:
        time=t_end
    if time < t_init:
        time=t_init

    q0_t = a_jnt0[0] + a_jnt0[1]*time + a_jnt0[2]*(time**2) + a_jnt0[3]*(time**3)
    q1_t = a_jnt1[0] + a_jnt1[1]*time + a_jnt1[2]*(time**2) + a_jnt1[3]*(time**3)
    q0_t_dot = a_jnt0[1] + 2*a_jnt0[2]*time + 3*a_jnt0[3]*(time**2)
    q1_t_dot = a_jnt1[1] + 2*a_jnt1[2]*time + 3*a_jnt1[3]*(time**2)

    # pd control
    d.ctrl[0] = -500 * (d.qpos[0] - q0_t) - 50*(d.qvel[0]-q0_t_dot)
    d.ctrl[1] = -500 * (d.qpos[1] - q1_t) - 50*(d.qvel[1]-q1_t_dot)

    # model-based control (feedback linearization)
    M = np.zeros((2, 2))
    mj.mj_fullM(m, d, M) # get mass matrix
    f0 = d.qfrc_bias[0]
    f1 = d.qfrc_bias[1]
    f = np.array([f0, f1])

    kp = 500
    kd = 2*np.sqrt(kp)
    pd_0 = -kp * (d.qpos[0] - q0_t) - kd*(d.qvel[0]-q0_t_dot)
    pd_1 = -kp * (d.qpos[1] - q1_t) - kd*(d.qvel[1]-q1_t_dot)
    pd_control = np.array([pd_0, pd_1])
    tau_M_pd_control = np.matmul(M, pd_control)
    tau = np.add(tau_M_pd_control, f)

    d.ctrl[0] = tau[0]
    d.ctrl[1] = tau[1]

    t.append(d.time)
    qact0.append(d.qpos[0])
    qref0.append(q0_t)
    qact1.append(d.qpos[1])
    qref1.append(q1_t)

def draw_circle(N, xc, yc, r):
    thetas = np.linspace(0, np.pi*4,N)
    xs = r*np.cos(thetas) + xc
    ys = r*np.sin(thetas) + yc
    return xs, ys

def generate_trajectory(t0, tf, q0, qf):
    tf_t0_3 = (tf - t0)**3
    a0 = qf*(t0**2)*(3*tf-t0) + q0*(tf**2)*(tf-3*t0)
    a0 = a0 / tf_t0_3

    a1 = 6 * t0 * tf * (q0 - qf)
    a1 = a1 / tf_t0_3

    a2 = 3 * (t0 + tf) * (qf - q0)
    a2 = a2 / tf_t0_3

    a3 = 2 * (q0 - qf)
    a3 = a3 / tf_t0_3

    return a0, a1, a2, a3


def main():
    m = mujoco.MjModel.from_xml_path(XML_PATH)
    d = mujoco.MjData(m)

    camera_name = "my_cam"
    camera_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    cam.fixedcamid = camera_id

    # Set initial state before launching viewer / registering controller
    d.qpos[0] = q0_init
    d.qpos[1] = q1_init
    mj.mj_forward(m, d)

    init_controller(m, d)
    mj.set_mjcb_control(controller)

    print("Launching MuJoCo Interactive Viewer...")
    print(f"Loading model from: {XML_PATH}")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        viewer.cam.type = cam.type
        viewer.cam.fixedcamid = cam.fixedcamid

        """xs, ys = draw_circle(N=2000, xc=0.75, yc=1, r=0.25)
        x_all = []
        y_all = []
        i = 0"""

        while viewer.is_running():
            step_start = time.time()

            """
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

            phi1 += dq[0]
            phi2 += dq[1]
            d.qpos[0] = phi1  # phi1_dot
            d.qpos[1] = phi2 # phi2_dot

            i += 1
            """
            if d.time>=t_end:                
                errors_0 = np.subtract(qref0, qact0)
                errors_1 = np.subtract(qref1, qact1)

                plt.figure(1)
                plt.subplot(2, 1, 1)
                plt.plot(t, errors_0, 'r-', label="Joint 0 Error")
                plt.ylabel("error (rad)")
                plt.legend()

                plt.subplot(2, 1, 2)
                plt.plot(t, errors_1, 'b-', label="Joint 1 Error")
                plt.xlabel("time (s)")
                plt.ylabel("error (rad)")
                plt.legend()

                plt.tight_layout()
                plt.savefig("error.png")
                print("Saved plot to error.png")
                plt.close()
                return
            
            mj.mj_step(m, d)
            
            viewer.sync()

            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()