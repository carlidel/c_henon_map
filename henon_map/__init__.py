import matplotlib.pyplot as plt
from numba import cuda, jit, njit
import numpy as np

from . import gpu_henon_core as gpu
from . import cpu_henon_core as cpu


@njit
def modulation(epsilon, n_elements, first_index=0):
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
        np.sum(coefficients * np.cos(modulations * k)) for k in range(first_index, first_index + n_elements)
    ])
    omega_x = 0.168 * 2 * np.pi * (1 + epsilon * omega_sum)
    omega_y = 0.201 * 2 * np.pi * (1 + epsilon * omega_sum)
    return omega_x, omega_y


class henon_map_2d(object):
    def __init__(self, x, p, epsilon, limit=100.0):
        """Init a 2d Henon Map object
        
        Parameters
        ----------
        object : self
            self
        x : ndarray
            x coordinates
        p : ndarray
            p coordinates
        epsilon : float
            modulation intensity
        limit : float, optional
            barrier limit in action, by default 100.0
        """        
        assert x.size == p.size
        self.x = x
        self.p = p
        self.x0 = x.copy()
        self.p0 = p.copy()
        self.epsilon = epsilon
        self.n_iterations = np.zeros(x.shape, dtype=np.int)
        self.limit = limit
        self.total_iters = 0

    def reset(self):
        """Resets the object.
        """        
        self.x = self.x0.copy()
        self.p = self.p0.copy()
        self.n_iterations = np.zeros(self.x.shape, dtype=np.int)
        self.total_iters = 0

    def compute(self, iterations_to_perform):
        """Computes a given number of iterations
        
        Parameters
        ----------
        iterations_to_perform : unsigned int
            iterations to perform
        
        Returns
        -------
        tuple
            (x, p, iterations)
        """        
        omega = modulation(self.epsilon, iterations_to_perform, self.total_iters)[0]
        self.x, self.p, self.n_iterations = cpu.henon_map_2D(
            self.x, self.p, self.n_iterations, self.limit, iterations_to_perform, omega)
        self.total_iters += iterations_to_perform
        return self.x, self.p, self.n_iterations

    def get_data(self):
        """Get the data
        
        Returns
        -------
        tuple
            (x, p, iterations)
        """        
        return self.x, self.p, self.n_iterations

    def get_coords(self):
        """Get coordinates
        
        Returns
        -------
        tuple
            (x, p)
        """        
        return self.x, self.p

    def get_iters(self):
        """Get times
        
        Returns
        -------
        ndarray
            times
        """        
        return self.n_iterations

    def get_filtered_coords(self):
        """Get filtered coordinates
        
        Returns
        -------
        tuple
            (x, p)
        """
        indexes = np.logical_and(self.x == 0.0, self.p == 0.0)
        return self.x[indexes], self.p[indexes]

    def get_filtered_iters(self):
        """Get filtered times
        
        Returns
        -------
        ndarray
            times
        """
        return self.n_iterations[self.n_iterations < self.total_iters]


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

        omega_x, omega_y = modulation(self.epsilon, sample_list[0])

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

        return np.transpose(np.asarray(self.container)) * self.dr

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

        return np.transpose(np.asarray(self.container))

    def get_data(self):
        """Get the data
        
        Returns
        -------
        ndarray
            the data
        """
        return np.transpose(np.asarray(self.container)) * self.dr


class gpu_full_track(object):
    def __init__(self, radius, alpha, theta1, theta2, epsilon, n_iterations):
        """init an henon optimized full tracker!
        
        Parameters
        ----------
        object : self
            self
        radius : ndarray        
            radiuses to consider (raw)
        alpha : ndarray
            alpha angles to consider (raw)
        theta1 : ndarray
            theta1 angles to consider (raw)
        theta2 : ndarray
            theta2 angles to consider (raw)
        epsilon : float
            intensity of modulation
        n_iterations : unsigned int
            number of iterations to track
        """
        assert alpha.size == theta1.size
        assert alpha.size == theta2.size
        assert alpha.size == radius.size

        # save data as members
        self.radius = radius
        self.alpha = alpha
        self.theta1 = theta1
        self.theta2 = theta2
        self.epsilon = epsilon
        self.n_iterations = n_iterations

        # make containers
        self.x = np.empty((n_iterations, alpha.size))
        self.px = np.empty((n_iterations, alpha.size))
        self.y = np.empty((n_iterations, alpha.size))
        self.py = np.empty((n_iterations, alpha.size))

        # load vectors to gpu
        self.d_alpha = cuda.to_device(self.alpha)
        self.d_theta1 = cuda.to_device(self.theta1)
        self.d_theta2 = cuda.to_device(self.theta2)
        self.d_radius = cuda.to_device(self.radius)

        self.d_x = cuda.device_array((n_iterations, alpha.size))
        self.d_px = cuda.device_array((n_iterations, alpha.size))
        self.d_y = cuda.device_array((n_iterations, alpha.size))
        self.d_py = cuda.device_array((n_iterations, alpha.size))

        # synchronize!
        cuda.synchronize()

    def compute(self):
        """Compute the tracking
        
        Returns
        -------
        tuple of 2D ndarray [n_iterations, n_samples]
            (radius, alpha, theta1, theta2)
        """
        threads_per_block = 1024
        blocks_per_grid = self.alpha.size // 1024 + 1

        omega_x, omega_y = modulation(self.epsilon, self.n_iterations)

        d_omega_x = cuda.to_device(omega_x)
        d_omega_y = cuda.to_device(omega_y)

        # Execution
        gpu.henon_map[blocks_per_grid, threads_per_block](
            self.d_radius, self.d_alpha, self.d_theta1, self.d_theta2,
            self.n_iterations, d_omega_x, d_omega_y,
            self.d_x, self.d_y, self.d_px, self.d_py
        )
        cuda.synchronize()

        self.d_x.copy_to_host(self.x)
        self.d_y.copy_to_host(self.y)
        self.d_px.copy_to_host(self.px)
        self.d_py.copy_to_host(self.py)

        return cartesian_to_polar_4d(self.x, self.y, self.px, self.py)

    def get_data(self):
        """Get the data
        
        Returns
        -------
        tuple of 2D ndarray [n_iterations, n_samples]
            (radius, alpha, theta1, theta2)
        """
        return cartesian_to_polar_4d(self.x, self.y, self.px, self.py)


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

        omega_x, omega_y = modulation(self.epsilon, sample_list[0])

        # Execution
        for sample in sample_list:
            self.step = cpu.henon_map(
                self.alpha, self.theta1, self.theta2,
                self.dr, self.step, self.limit,
                sample, omega_x, omega_y)
            self.container.append(self.step.copy())

        return np.transpose(np.asarray(self.container)) * self.dr

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

        return np.transpose(np.asarray(self.container))

    def advanced_dummy_compute(self, sample_list):
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
            self.step = cpu.dummy_map(
                self.alpha, self.theta1, self.theta2, self.dr, self.step, sample)
            self.container.append(self.step.copy())

        return np.transpose(np.asarray(self.container))

    def get_data(self):
        """Get the data
        
        Returns
        -------
        ndarray
            the data
        """
        return np.transpose(np.asarray(self.container)) * self.dr


class cpu_full_track(object):
    def __init__(self, radius, alpha, theta1, theta2, epsilon, n_iterations):
        """init an henon optimized full tracker!
        
        Parameters
        ----------
        object : self
            self
        radius : ndarray        
            radiuses to consider (raw)
        alpha : ndarray
            alpha angles to consider (raw)
        theta1 : ndarray
            theta1 angles to consider (raw)
        theta2 : ndarray
            theta2 angles to consider (raw)
        epsilon : float
            intensity of modulation
        n_iterations : unsigned int
            number of iterations to track
        """
        assert alpha.size == theta1.size
        assert alpha.size == theta2.size
        assert alpha.size == radius.size

        # save data as members
        self.radius = radius
        self.alpha = alpha
        self.theta1 = theta1
        self.theta2 = theta2
        self.epsilon = epsilon
        self.n_iterations = n_iterations

        # make containers
        self.x = np.empty((n_iterations, alpha.size))
        self.px = np.empty((n_iterations, alpha.size))
        self.y = np.empty((n_iterations, alpha.size))
        self.py = np.empty((n_iterations, alpha.size))

    def compute(self):
        """Compute the tracking
        
        Returns
        -------
        tuple of 2D ndarray [n_iterations, n_samples]
            (radius, alpha, theta1, theta2)
        """
        omega_x, omega_y = modulation(self.epsilon, self.n_iterations)
        # Execution
        self.x, self.y, self.px, self.py = cpu.henon_full_track(
            self.radius, self.alpha, self.theta1, self.theta2,
            self.n_iterations, omega_x, omega_y
        )
        return cartesian_to_polar_4d(self.x, self.y, self.px, self.py)

    def get_data(self):
        """Get the data
        
        Returns
        -------
        tuple of 2D ndarray [n_iterations, n_samples]
            (radius, alpha, theta1, theta2)
        """
        return cartesian_to_polar_4d(self.x, self.y, self.px, self.py)


class cpu_partial_track(object):
    def __init__(self, radius, alpha, theta1, theta2, epsilon):
        """init an henon optimized full tracker!
        
        Parameters
        ----------
        object : self
            self
        radius : ndarray        
            radiuses to consider (raw)
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
        assert alpha.size == radius.size

        # save data as members
        self.r = radius
        self.alpha = alpha
        self.theta1 = theta1
        self.theta2 = theta2
        self.r_0 = radius.copy()
        self.alpha_0 = alpha.copy()
        self.theta1_0 = theta1.copy()
        self.theta2_0 = theta2.copy()
        self.epsilon = epsilon
        self.total_iters = 0
        self.limit = 100.0

        # make containers
        self.step = np.zeros((alpha.size), dtype=np.int)

        self.x = np.empty(alpha.size)
        self.px = np.empty(alpha.size)
        self.y = np.empty(alpha.size)
        self.py = np.empty(alpha.size)

        self.x, self.y, self.px, self.py = cpu.polar_to_cartesian(
            self.r_0, self.alpha_0, self.theta1_0, self.theta2_0)

    def compute(self, n_iterations):
        """Compute the tracking
        
        Returns
        -------
        tuple of 2D ndarray [n_iterations, n_samples]
            (radius, alpha, theta1, theta2, steps)
        """
        omega_x, omega_y = modulation(self.epsilon, n_iterations, self.total_iters)
        # Execution
        self.x, self.y, self.px, self.py, self.step = cpu.henon_partial_track(
            self.x, self.y, self.px, self.py, self.step, self.limit,
            n_iterations, omega_x, omega_y
        )
        self.total_iters += n_iterations
        self.r, self.alpha, self.theta1, self.theta2 = cpu.cartesian_to_polar(self.x, self.y, self.px, self.py)
        return self.r, self.alpha, self.theta1, self.theta2, self.step

    def get_data(self):
        """Get the data
        
        Returns
        -------
        tuple
            (radius, alpha, theta1, theta2)
        """
        return self.r, self.alpha, self.theta1, self.theta2, self.step

    def get_radiuses(self):
        return self.r

    def get_filtered_radiuses(self):
        return self.r[self.r != 0.0]

    def get_times(self):
        return self.step

    def get_action(self):
        return np.power(self.r, 2)
    
    def get_filtered_action(self):
        return np.power(self.r[self.r != 0.0], 2)

    def reset(self):
        self.r = self.r_0
        self.alpha = self.alpha_0
        self.theta1 = self.theta1_0
        self.theta2 = self.theta2_0
        self.step = np.zeros((self.alpha.size), dtype=np.int)
        self.x, self.y, self.px, self.py = cpu.polar_to_cartesian(
            self.r_0, self.alpha_0, self.theta1_0, self.theta2_0)
        self.total_iters = 0


def henon_single_call(*args, **kwargs):
    """Henon_map single call
    
    Parameters
    ----------
    alpha : float
        alpha angle
    theta1 : float
        theta call
    theta2 : float
        theta angle
    dr : float
        step
    epsilon : float
        intensity
    n_iterations : unsigned int
        number of iterations
    
    Returns
    -------
    float
        the radius
    """
    alpha, theta1, theta2, dr, epsilon, n_iterations = args
    omega_x, omega_y = modulation(epsilon, n_iterations)
    return dr * np.transpose(
        np.asarray(
            cpu.henon_map(
                np.asarray([alpha]),
                np.asarray([theta1]),
                np.asarray([theta2]),
                dr,
                np.zeros((1), dtype=np.int),
                100.0,
                n_iterations,
                omega_x, omega_y
            )
        )
    )


def advanced_dummy_call(alpha, theta1, theta2, r):
    """dummy single call
    
    Parameters
    ----------
    alpha : ndarray
        alpha angle
    theta1 : ndarray
        theta call
    theta2 : ndarray
        theta angle
    r : float
        radius
    
    Returns
    -------
    float
        the radius
    """
    return cpu.advanced_dummy_map(np.asarray(alpha), np.asarray(theta1), np.asarray(theta2), r)


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
