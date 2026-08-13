import numpy as np

class Accelerometer:
    def __init__(self, n, sigma, random_walk):
        self.n = n
        self.sigma = sigma
        self.random_walk = random_walk
        self.bias = np.zeros(n)

    def sample(self, true_acceleration: np.ndarray, dt: float) -> np.ndarray:
        """
        Generates a noisy acceleration measurement.
        :param true_acceleration: (3,) numpy array of true acceleration in body frame.
        :return: (3,) numpy array of noisy acceleration measurement in body frame.
        """
        assert true_acceleration.shape[0] == self.n

        white_noise = np.random.normal(0, self.sigma, size=self.n)
        bias_drift = np.random.normal(0, np.sqrt(self.random_walk*dt), size=self.n)
        self.bias += bias_drift
        return true_acceleration + white_noise + self.bias

class Gyroscope:
    def __init__(self, n, sigma, random_walk):
        self.n = n
        self.sigma = sigma
        self.random_walk = random_walk
        self.bias = np.zeros(n)

    def sample(self, true_angular_velocity: np.ndarray, dt: float) -> np.ndarray:
        """
        Generates a noisy angular velocity measurement.
        :param true_angular_velocity: (3,) numpy array of true angular velocity in body frame.
        :return: (3,) numpy array of noisy angular velocity measurement in body frame.
        """
        assert len(true_angular_velocity) == self.n

        white_noise = np.random.normal(0, self.sigma, size=self.n)
        bias_drift = np.random.normal(0, np.sqrt(self.random_walk*dt), size=self.n)
        self.bias += bias_drift
        return true_angular_velocity + white_noise + self.bias

class Magnetometer:
    def __init__(self, n, sigma, bias_drift):
        self.n = n
        self.sigma = sigma
        self.bias_drift = bias_drift
        self.bias = np.zeros(n)

    def sample(self, true_magnetic_field: np.ndarray) -> np.ndarray:
        """
        Generates a noisy magnetic field measurement.
        :param true_magnetic_field: (3,) numpy array of true magnetic field in body frame.
        :return: (3,) numpy array of noisy magnetic field measurement in body frame.
        """

    def sample(self, true_magnetic_field: np.ndarray) -> np.ndarray:
        """
        Generates a noisy magnetic field measurement.
        :param true_magnetic_field: (3,) numpy array of true magnetic field in body frame.
        :return: (3,) numpy array of noisy magnetic field measurement in body frame.
        """
        noise = np.random.normal(0, self.sigma, size=true_magnetic_field.shape)
        return true_magnetic_field + noise
    
class GPS:
    def __init__(self):
        self.sigma = 0

    def sample(self, true_position: np.ndarray) -> np.ndarray:
        """
        Generates a noisy position measurement.
        :param true_position: (3,) numpy array of true position in world frame.
        :return: (3,) numpy array of noisy position measurement in world frame.
        """
        noise = np.random.normal(0, self.sigma, size=true_position.shape)
        return true_position + noise
    