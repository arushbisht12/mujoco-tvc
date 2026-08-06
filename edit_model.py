import os 
import mujoco as mj
import mujoco.viewer

# Get absolute path to XML file
XML_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "ball.xml"))

def load_model():
    if not os.path.exists(XML_PATH):
        raise FileNotFoundError(f"XML file not found at: {XML_PATH}")
    m = mujoco.MjModel.from_xml_path(XML_PATH)
    d = mujoco.MjData(m)
    return m, d

def main():

    """d.qpos[0] = 0
    d.qpos[1] = 1.57"""

    print("Launching MuJoCo Interactive Viewer...")
    mujoco.viewer.launch(loader=load_model)

if __name__ == "__main__":
    main()

        
    