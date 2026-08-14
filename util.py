import numpy as np
from pyquaternion import Quaternion

def skewSymmetric(v):
  return np.array([[0.0, -v[2].item(), v[1].item()],
                  [v[2].item(), 0.0, -v[0].item()],
                  [-v[1].item(), v[0].item(), 0.0]]) 
  

"""def quatToMatrix(q): # body to inertial
  return 2.0*np.outer(q, q.vector) \
           + np.identity(3)*(q.scalar**2 - q.vector.dot(q.vector)) \
           + 2*q.scalar*skewSymmetric(q.vector)"""
           
def quatToMatrix(q): # body to inertial
    q = Quaternion(q)
    return 2.0*np.outer(q, q.vector) \
            + np.identity(3)*(q.scalar**2 - q.vector.dot(q.vector)) \
            + 2*q.scalar*skewSymmetric(q.vector)

def quat_multiply(q, r):
  """Multiplies two quaternions q and r."""
  w1, x1, y1, z1 = q
  w2, x2, y2, z2 = r
  return np.array([
      w1*w2 - x1*x2 - y1*y2 - z1*z2,
      w1*x2 + x1*w2 + y1*z2 - z1*y2,
      w1*y2 - x1*z2 + y1*w2 + z1*x2,
      w1*z2 + x1*y2 - y1*x2 + z1*w2
  ], dtype=float)
  
def quat_to_euler(q):
    """
    Converts a quaternion [w, x, y, z] to Euler angles (Roll, Pitch, Yaw).
    Returns angles in radians.
    """
    w, x, y, z = q
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x**2 + y**2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    # Handle gimbal lock (clamping to -90 to 90 degrees)
    if abs(sinp) >= 1:
        pitch = np.sign(sinp) * (np.pi / 2)
    else:
        pitch = np.arcsin(sinp)
        
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y**2 + z**2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return np.array([roll, pitch, yaw])