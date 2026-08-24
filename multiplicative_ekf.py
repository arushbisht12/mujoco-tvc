import numpy as np
from numpy import linalg as la
from util import skewSymmetric, quat_multiply, quat_to_euler, quat_rotation_matrix
from pyquaternion import Quaternion

class MEKF:
    def __init__(self, initial_q, initial_vel, initial_pos,
                 gyro_cov, accel_cov,         # Values usually found in IMU datasheet (units like rad/s/sqrt(Hz))
                 gyro_bias_cov, accel_bias_cov,
                 gps_cov=1.0, P_initial=None) -> None:
        initial_gyr_b = np.zeros(3)
        initial_acc_b = np.zeros(3)
        self._x = np.concatenate((np.asarray(initial_q).flatten(),
                                  np.asarray(initial_vel).flatten(),
                                  np.asarray(initial_pos).flatten(),
                                  initial_gyr_b,
                                  initial_acc_b)).reshape((16,1)) # handles 1D and 2D initial inputs
        
        #self._error_x = np.zeros(15).reshape(-1,1)     # error state vector is always 0 in predict step so not needed to be stored
        if P_initial is None:
            self._P = np.identity(15, dtype=float)
        else:
            self._P = np.array(P_initial, dtype=float)
        
        # PARAMETERS
        self.dt = 0.01  #PLACEHOLDER
        self.g_const = 9.80665
        self.theta_const = 0
        self.g_vector = np.array([0,0,1]).reshape((3,1))
        self.north_vector = np.array([0, np.cos(self.theta_const), -np.sin(self.theta_const)]).reshape([3,1])
        
        self.gyro_cov_mat = gyro_cov * np.identity(3, dtype=float)
        self.accel_cov_mat = accel_cov * np.identity(3, dtype=float)
        self.gyro_bias_cov_mat = gyro_bias_cov * np.identity(3, dtype=float)
        self.accel_bias_cov_mat = accel_bias_cov * np.identity(3, dtype=float)
        self.gps_cov_mat = gps_cov * np.identity(3, dtype=float)


    def prediction(self, dt, gyr=None, acc=None):
        # subtract biases from sensor measurements
        if gyr is None:
            gyr = np.zeros(3).reshape(-1,1)
        else: 
            gyr = gyr - self.gyr_b.reshape(-1,1)
        if acc is None:
            acc = np.zeros(3).reshape(-1,1)
        else:
            acc = acc - self.acc_b.reshape(-1,1)
        
        # predict nominal values
        self._x[0:10] = self.f(gyr, acc, dt)
        
        # predict error values
        process_model = self.phi(gyr, acc, dt)
        Q = self.process_covariance(dt)
        self._P = process_model @ self._P @ process_model.T + Q
        
    def correction(self, dt: float, acc=None, mag=None, gps=None, lid=None, bar=None) -> None:
        if acc is None:
            acc_norm = np.zeros(3).reshape(-1, 1)
        else:
            acc_norm = acc / la.vector_norm(acc)
        if mag is None:
            mag_norm = np.zeros(3).reshape(-1, 1)
        else:
            mag_norm = mag / la.vector_norm(mag)
        if gps is None:
            gps = np.zeros(3).reshape(-1, 1)
        if lid is None:
            lid = np.zeros(1).reshape(-1, 1)
        if bar is None:
            bar = np.zeros(1).reshape(-1, 1)
            
        
        # Kalman gain
        H_k = self.H()
        P_Ht = self._P @ H_k.T
        # Calculate P Ht used twice
        # S = H P Ht + R
        S = H_k @ P_Ht + self.measurement_covariance()
        # K = P Ht S^-1
        K_k = P_Ht @ np.linalg.solve(S, np.eye(S.shape[0]))
        
        # A posteriori covariance
        self._P = (np.identity(15, dtype=float) - K_k @ H_k) @ self._P
        
        # A posteriori ERROR state
        z = np.concatenate((acc_norm, mag_norm, gps, lid, bar)).reshape((11,1))
        error_state = K_k @ (z - self.h())
        
        
        # Fold error state back into nominal states
        small_angle_approx_q = np.concatenate(([1.0], 0.5 * error_state[0:3].ravel())).reshape(-1,1)
        new_q = quat_multiply(self.q, small_angle_approx_q)
        self._x[0:4] = new_q / np.linalg.norm(new_q)
        self._x[4:7] = self.vel.reshape(-1,1) + error_state[3:6]
        self._x[7:10] = self.pos.reshape(-1,1) + error_state[6:9]
        self._x[10:13] += error_state[9:12]
        self._x[13:16] += error_state[12:15]
        
       
    #HELPER
        
    def f(self, gyr, acc, dt: float) -> np.ndarray:
        q_prev = self._x[0:4]
        vel_prev = self._x[4:7]
        pos_prev = self._x[7:10]
        
        q_new = (np.eye(4) + 0.5 * dt * self.omega(gyr)).dot(q_prev) # switch to @
        q_new = q_new / np.linalg.norm(q_new) # normalize to prevent drift
        #use q_old or q_new?
        vel_new = vel_prev + self.rotation_body_to_earth(q_new).dot(acc) * dt # switch to @
        vel_new[2] -= self.g_const * dt # to account for graivty acceleration
        pos_new = pos_prev + vel_new * dt
        
        return np.concatenate((q_new, vel_new, pos_new)).reshape((10,1))
    
    # state transition matrix
    def phi(self, gyr, acc, dt) -> np.ndarray:
        G = np.zeros(shape=(15, 15), dtype=float)
        
        G[0:3, 0:3] = -skewSymmetric(gyr)
        G[3:6, 0:3] = -quat_rotation_matrix(self.q) @ skewSymmetric(acc)
        G[3:6, 12:15] = -quat_rotation_matrix(self.q)
        G[6:9, 3:6] = np.identity(3, dtype=float)
        G[0:3, 9:12] = -np.identity(3, dtype=float)
        
        return np.identity(15, dtype=float) + G * dt 
    
    def process_covariance(self, dt):
        Q = np.zeros((15, 15), dtype=float)
        
        dt2 = dt**2
        dt3 = dt**3
        dt4 = dt**4
        dt5 = dt**5

        Q[0:3, 0:3] = self.gyro_cov_mat * dt + self.gyro_bias_cov_mat * (dt3 / 3.0)
        Q[9:12, 0:3] = -self.gyro_bias_cov_mat * (dt2 / 2.0)
        Q[0:3, 9:12] = Q[9:12, 0:3]
        Q[9:12, 9:12] = self.gyro_bias_cov_mat * dt

        Q[3:6, 3:6] = self.accel_cov_mat * dt + self.accel_bias_cov_mat * (dt3 / 3.0)
        
        Q[6:9, 3:6] = self.accel_cov_mat * (dt2 / 2.0) + self.accel_bias_cov_mat * (dt4 / 8.0)
        Q[3:6, 6:9] = Q[6:9, 3:6]
        
        Q[12:15, 3:6] = -self.accel_bias_cov_mat * (dt2 / 2.0)
        Q[3:6, 12:15] = Q[12:15, 3:6]
        
        Q[6:9, 6:9] = self.accel_cov_mat * (dt3 / 3.0) + self.accel_bias_cov_mat * (dt5 / 20.0)
        
        Q[12:15, 6:9] = -self.accel_bias_cov_mat * (dt3 / 6.0)
        Q[6:9, 12:15] = Q[12:15, 6:9]
        
        Q[12:15, 12:15] = self.accel_bias_cov_mat * dt

        return Q
        
    def h(self) -> np.ndarray:
        acc_ex = self.rotation_earth_to_body(self.q) @ self.g_vector
        mag_ex = self.rotation_earth_to_body(self.q) @ self.north_vector
        gps_ex = self.pos.reshape((3,1))
        lid_ex = np.array([self.pos[2]]).reshape((1,1))
        bar_ex = np.array([self.pos[2]]).reshape((1,1))
        
        return np.concatenate((acc_ex,mag_ex,gps_ex,lid_ex,bar_ex))
    
    def H(self) -> np.ndarray: 
        H = np.zeros(shape = (11,15), dtype=float)
        
        H[0:3, 0:3] = skewSymmetric( self.rotation_earth_to_body(self.q) @ self.g_vector)
        # H[0:3, 12:15] = np.identity(3, dtype=float) # REMOVED: acc_b is in m/s^2, acc_norm is unitless
        H[3:6, 0:3] = skewSymmetric( self.rotation_earth_to_body(self.q) @ self.north_vector)
        # H[3:6, 12:15] = np.identity(3, dtype=float) # BUG: Magnetometer does not depend on accelerometer bias!
        H[6:9, 6:9] = np.identity(3, dtype=float)
        H[9, 8] = 1.0
        H[10, 8] = 1.0
        
        return H
    
    def measurement_covariance(self):
        R = np.identity(11, dtype=float)
        R[6:9, 6:9] = self.gps_cov_mat
        return R
        
        
    @property
    def q(self) -> np.ndarray:
        return self._x[0:4,0]  
    
    @property
    def vel(self) -> np.ndarray:
        return self._x[4:7,0]
    
    @property
    def pos(self) -> np.ndarray:
        return self._x[7:10,0]
    
    @property 
    def gyr_b(self) -> np.ndarray:
        return self._x[10:13,0]
    
    @property 
    def acc_b(self) -> np.ndarray:
        return self._x[13:16,0]
    
    @property
    def mean(self) -> np.ndarray:
        return np.concatenate((quat_to_euler(self._x[0:4,0]),(self._x[4:,0])))
    
    @property
    def cov(self) -> np.ndarray:
        return self._P
    
    
    
    
    def omega(self, gyr) -> np.ndarray: 
        gp, gq, gr = np.asarray(gyr).reshape(3,)
        return np.array([
            [0.0, -gp, -gq, -gr],
            [gp,  0.0,  gr, -gq],
            [gq, -gr,  0.0, gp], #mistake in lorenzo's omega
            [gr,  gq, -gp,  0.0]
        ])
         
    def rotation_body_to_earth(self, q: np.ndarray) -> np.ndarray:
        a, b, c, d = np.asarray(q).reshape(4,)
        return np.array([
            [a*a + b*b - c*c - d*d,  -2*a*d + 2*b*c,         2*a*c + 2*b*d],
            [2*a*d + 2*b*c,          a*a - b*b + c*c - d*d,  -2*a*b + 2*c*d],
            [-2*a*c + 2*b*d,         2*a*b + 2*c*d,          a*a - b*b - c*c + d*d]
    ])
        
    def rotation_earth_to_body(self, q:np.ndarray) -> np.ndarray:
        return self.rotation_body_to_earth(q).T
    
    
    