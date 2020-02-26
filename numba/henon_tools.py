from numba import cuda
import numpy as np

import gpu_henon_core as gpu
import cpu_henon_core as cpu


class gpu_radial_scan(object):
    def __init__(self, dr, alpha, theta1, theta2, epsilon):
        """init an henon optimized radial tracker!
        
        Parameters
        ----------
        object : self
            self
        dr : float
            radial step
        alpha : ndarray
            alpha angles to consider (raw)
        theta1 : ndarray
            theta1 angles to consider (raw)
        theta2 : ndarray
            theta2 angles to consider (raw)
        epsilon : float
            intensity of modulation
        """
        assert alpha.size == theta1.size
        assert alpha.size == theta2.size

        # save data as members
        self.dr = dr
        self.alpha = alpha
        self.theta1 = theta1
        self.theta2 = theta2
        self.epsilon = epsilon
        self.limit = 100.0

        # prepare data
        self.step = np.zeros(alpha.shape, dtype=np.int)

        # make container
        self.container = []

        # load vectors to gpu
        self.d_alpha = cuda.to_device(self.alpha)
        self.d_theta1 = cuda.to_device(self.theta1)
        self.d_theta2 = cuda.to_device(self.theta2)
        self.d_step = cuda.to_device(self.step)
        # synchronize!
        cuda.synchronize()

    def reset(self):
        """Resets the engine.
        """
        self.container = []
        self.step = np.zeros(self.alpha.shape, dtype=np.int)
        self.d_step = cuda.to_device(self.step)
        # synchronize!
        cuda.synchronize()

    def compute(self, sample_list):
        """Compute the tracking
        
        Parameters
        ----------
        sample_list : ndarray
            iterations to consider
        
        Returns
        -------
        ndarray
            radius scan results
        """
        threads_per_block = 512
        blocks_per_grid = self.step.size // 512 + 1
        # Sanity check
        assert blocks_per_grid * threads_per_block > self.alpha.size
        for i in range(1, len(sample_list)):
            assert sample_list[i] <= sample_list[i - 1]

        # Modulation
        coefficients = np.array([1.000e-4,
                                 0.218e-4,
                                 0.708e-4,
                                 0.254e-4,
                                 0.100e-4,
                                 0.078e-4,
                                 0.218e-4])
        modulations = np.array([1 * (2 * np.pi / 868.12),
                                2 * (2 * np.pi / 868.12),
                                3 * (2 * np.pi / 868.12),
                                6 * (2 * np.pi / 868.12),
                                7 * (2 * np.pi / 868.12),
                                10 * (2 * np.pi / 868.12),
                                12 * (2 * np.pi / 868.12)])
        omega_sum = np.array([
            np.sum(coefficients * np.cos(modulations * k)) for k in range(sample_list[0])
        ])
        omega_x = 0.168 * 2 * np.pi * (1 + self.epsilon * omega_sum)
        omega_y = 0.201 * 2 * np.pi * (1 + self.epsilon * omega_sum)

        d_omega_x = cuda.to_device(omega_x)
        d_omega_y = cuda.to_device(omega_y)

        # Execution
        for sample in sample_list:
            gpu.henon_map[blocks_per_grid, threads_per_block](
                self.d_alpha, self.d_theta1, self.d_theta2,
                self.dr, self.d_step, self.limit,
                sample, d_omega_x, d_omega_y)
            cuda.synchronize()
            self.d_step.copy_to_host(self.step)
            self.container.append(self.step.copy())

        return np.asarray(self.container)
    
    def dummy_compute(self, sample_list):
        """performs a dummy computation
        
        Parameters
        ----------
        sample_list : ndarray
            iterations to consider

        Returns
        -------
        ndarray
            radius dummy results
        """
        # Execution
        for sample in sample_list:
            gpu.dummy_map[self.step.size // 512 + 1, 512](
                self.d_step, sample)
            cuda.synchronize()
            self.d_step.copy_to_host(self.step)
            self.container.append(self.step.copy())

        return np.asarray(self.container)

    def get_data(self):
        """Get the data
        
        Returns
        -------
        ndarray
            the data
        """
        return np.asarray(self.container)



class cpu_radial_scan(object):
    def __init__(self, dr, alpha, theta1, theta2, epsilon):
        """init an henon optimized radial tracker!
        
        Parameters
        ----------
        object : self
            self
        dr : float
            radial step
        alpha : ndarray
            alpha angles to consider (raw)
        theta1 : ndarray
            theta1 angles to consider (raw)
        theta2 : ndarray
            theta2 angles to consider (raw)
        epsilon : float
            intensity of modulation
        """
        assert alpha.size == theta1.size
        assert alpha.size == theta2.size

        # save data as members
        self.dr = dr
        self.alpha = alpha
        self.theta1 = theta1
        self.theta2 = theta2
        self.epsilon = epsilon
        self.limit = 100.0

        # prepare data
        self.step = np.zeros(alpha.shape, dtype=np.int)

        # make container
        self.container = []

    def reset(self):
        """Resets the engine.
        """
        self.container = []
        self.step = np.zeros(self.alpha.shape, dtype=np.int)

    def compute(self, sample_list):
        """Compute the tracking
        
        Parameters
        ----------
        sample_list : ndarray
            iterations to consider
        
        Returns
        -------
        ndarray
            radius scan results
        """
        # Sanity check
        for i in range(1, len(sample_list)):
            assert sample_list[i] <= sample_list[i - 1]

        # Modulation
        coefficients = np.array([1.000e-4,
                                 0.218e-4,
                                 0.708e-4,
                                 0.254e-4,
                                 0.100e-4,
                                 0.078e-4,
                                 0.218e-4])
        modulations = np.array([1 * (2 * np.pi / 868.12),
                                2 * (2 * np.pi / 868.12),
                                3 * (2 * np.pi / 868.12),
                                6 * (2 * np.pi / 868.12),
                                7 * (2 * np.pi / 868.12),
                                10 * (2 * np.pi / 868.12),
                                12 * (2 * np.pi / 868.12)])
        omega_sum = np.array([
            np.sum(coefficients * np.cos(modulations * k)) for k in range(sample_list[0])
        ])
        omega_x = 0.168 * 2 * np.pi * (1 + self.epsilon * omega_sum)
        omega_y = 0.201 * 2 * np.pi * (1 + self.epsilon * omega_sum)

        # Execution
        for sample in sample_list:
            self.step = cpu.henon_map(
                self.alpha, self.theta1, self.theta2,
                self.dr, self.step, self.limit,
                sample, omega_x, omega_y)
            self.container.append(self.step.copy())

        return np.asarray(self.container)

    def dummy_compute(self, sample_list):
        """performs a dummy computation
        
        Parameters
        ----------
        sample_list : ndarray
            iterations to consider

        Returns
        -------
        ndarray
            radius dummy results
        """       
        # Execution
        for sample in sample_list:
            self.step = cpu.dummy_map(self.step, sample)
            self.container.append(self.step.copy())

        return np.asarray(self.container)

    def get_data(self):
        """Get the data
        
        Returns
        -------
        ndarray
            the data
        """
        return np.asarray(self.container)


def cartesian_to_polar_4d(x, y, px, py):
    """Convert a 4d cartesian point to a 4d polar variable point.
    
    Parameters
    ----------
    x : ndarray
        ipse dixit
    y : ndarray
        ipse dixit
    px : ndarray
        ipse dixit
    py : ndarray
        ipse dixit
    
    Returns
    -------
    tuple of ndarray
        (r, alpha, theta1, theta2)
    """
    r = np.sqrt(np.power(x, 2) + np.power(y, 2) +
                np.power(px, 2) + np.power(py, 2))
    theta1 = np.arctan2(px, x) + np.pi
    theta2 = np.arctan2(py, y) + np.pi
    alpha = np.arctan2(np.sqrt(y * y + py * py),
                       np.sqrt(x * x + px * px)) + np.pi
    return r, alpha, theta1, theta2
