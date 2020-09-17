import matplotlib.pyplot as plt
from numba import cuda, jit, njit, prange
import numpy as np
from tqdm import tqdm
import pickle
import time

from . import gpu_henon_core as gpu
from . import cpu_henon_core as cpu

from .cpu_henon_core import recursive_accumulation as cpu_accumulate_and_return

def polar_to_cartesian(radius, alpha, theta1, theta2):
    return cpu.polar_to_cartesian(radius, alpha, theta1, theta2)


def cartesian_to_polar(x, px, y, py):
    return cpu.cartesian_to_polar(x, px, y, py)


@njit
def modulation(epsilon, n_elements, first_index=0):
    """Generates a modulation
    
    Parameters
    ----------
    epsilon : float
        intensity of modulation
    n_elements : float
        number of elements
    first_index : int, optional
        starting point of the modulation, by default 0
    
    Returns
    -------
    tuple of ndarray
        (omega_x, omega_y)
    """    
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


class partial_track(object):
    """Kinda of a deprecated method. This class is meant to do a partial tracking (i.e. only last step is considered) of given initial condistions.
    """
    def __init__(self):
        pass

    def compute(self, n_iterations):
        pass

    def reset(self):
        pass

    def get_data(self):
        """Get the data
        
        Returns
        -------
        tuple
            (radius, alpha, theta1, theta2)
        """
        return self.r, self.alpha, self.theta1, self.theta2, self.step

    def get_cartesian_data(self):
        x, px, y, py = polar_to_cartesian(self.r, self.alpha, self.theta1, self.theta2)
        return x, px, y, py, self.step

    def get_radiuses(self):
        return self.r

    def get_filtered_radiuses(self):
        return self.r[self.r != 0.0]

    def get_times(self):
        return self.step

    def get_action(self):
        return np.power(self.r, 2) / 2

    def get_filtered_action(self):
        return np.power(self.r[self.r != 0.0], 2) / 2

    def get_survival_count(self):
        return np.count_nonzero(self.r != 0.0)

    def get_total_count(self):
        return self.r.size

    def get_survival_rate(self):
        return np.count_nonzero(self.r != 0.0) / self.r.size

    @staticmethod
    def generate_instance(radius, alpha, theta1, theta2, epsilon, cuda_device=None):
        """Generate an instance of the engine.
        
        Parameters
        ----------
        radius : ndarray
            array of radiuses to consider
        alpha : ndarray
            array of initial alphas
        theta1 : ndarray
            array of initial theta1
        theta2 : ndarray
            array of initial theta2
        epsilon : float
            modulation intensity
        
        Returns
        -------
        class instance
            optimized class instance
        """        
        if cuda_device == None:
            cuda_device = cuda.is_available()
        if cuda_device:
            return gpu_partial_track(radius, alpha, theta1, theta2, epsilon)
        else:
            return cpu_partial_track(radius, alpha, theta1, theta2, epsilon)


class cpu_partial_track(partial_track):
    def __init__(self, radius, alpha, theta1, theta2, epsilon):
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
        self.limit = 1.0

        # make containers
        self.step = np.zeros((alpha.size), dtype=np.int)

        self.x = np.empty(alpha.size)
        self.px = np.empty(alpha.size)
        self.y = np.empty(alpha.size)
        self.py = np.empty(alpha.size)

        self.x, self.px, self.y, self.py = cpu.polar_to_cartesian(
            self.r_0, self.alpha_0, self.theta1_0, self.theta2_0)

    def compute(self, n_iterations):
        """Compute the tracking
        
        Returns
        -------
        tuple of ndarray [n_elements]
            (radius, alpha, theta1, theta2, steps)
        """
        omega_x, omega_y = modulation(
            self.epsilon, n_iterations, self.total_iters)
        # Execution
        self.x, self.px, self.y, self.py, self.step = cpu.henon_partial_track(
            self.x, self.px, self.y, self.py, self.step, self.limit,
            n_iterations, omega_x, omega_y
        )
        self.total_iters += n_iterations
        self.r, self.alpha, self.theta1, self.theta2 = cpu.cartesian_to_polar(
            self.x, self.px, self.y, self.py)
        return self.x, self.px, self.y, self.py, self.step
        #return self.r, self.alpha, self.theta1, self.theta2, self.step

    def reset(self):
        """Resets the engine
        """        
        self.r = self.r_0
        self.alpha = self.alpha_0
        self.theta1 = self.theta1_0
        self.theta2 = self.theta2_0
        self.step = np.zeros((self.alpha.size), dtype=np.int)
        self.x, self.px, self.y, self.py = cpu.polar_to_cartesian(
            self.r_0, self.alpha_0, self.theta1_0, self.theta2_0)
        self.total_iters = 0


class gpu_partial_track(partial_track):
    def __init__(self, radius, alpha, theta1, theta2, epsilon):
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
        self.limit = 1.0

        # make containers
        self.step = np.zeros((alpha.size), dtype=np.int)

        self.x = np.empty(alpha.size)
        self.px = np.empty(alpha.size)
        self.y = np.empty(alpha.size)
        self.py = np.empty(alpha.size)

        self.x, self.px, self.y, self.py = cpu.polar_to_cartesian(
            self.r_0, self.alpha_0, self.theta1_0, self.theta2_0)

    def compute(self, n_iterations):
        """Compute the tracking
        
        Returns
        -------
        tuple of ndarray [n_elements]
            (radius, alpha, theta1, theta2, steps)
        """
        threads_per_block = 512
        blocks_per_grid = self.alpha.size // 512 + 1

        # load to GPU
        d_x = cuda.to_device(self.x)
        d_y = cuda.to_device(self.y)
        d_px = cuda.to_device(self.px)
        d_py = cuda.to_device(self.py)
        d_step = cuda.to_device(self.step)

        omega_x, omega_y = modulation(
            self.epsilon, n_iterations, self.total_iters)
        d_omega_x = cuda.to_device(omega_x)
        d_omega_y = cuda.to_device(omega_y)

        # Execution
        gpu.henon_partial_track[blocks_per_grid, threads_per_block](
            d_x, d_px, d_y, d_py, d_step, self.limit,
            n_iterations, d_omega_x, d_omega_y
        )
        self.total_iters += n_iterations

        d_x.copy_to_host(self.x)
        d_y.copy_to_host(self.y)
        d_px.copy_to_host(self.px)
        d_py.copy_to_host(self.py)
        d_step.copy_to_host(self.step)
        
        self.r, self.alpha, self.theta1, self.theta2 = cpu.cartesian_to_polar(self.x, self.px, self.y, self.py)

        return self.x, self.px, self.y, self.py, self.step
        #return self.r, self.alpha, self.theta1, self.theta2, self.step

    def reset(self):
        """Resets the engine
        """        
        self.r = self.r_0
        self.alpha = self.alpha_0

        self.theta1 = self.theta1_0
        self.theta2 = self.theta2_0
        self.step = np.zeros((self.alpha.size), dtype=np.int)

        self.x, self.px, self.y, self.py = gpu.actual_polar_to_cartesian(
            self.r_0, self.alpha_0, self.theta1_0, self.theta2_0)

        self.total_iters = 0


class uniform_scan(object):
    """With this class we can easly scan a uniform 4D cube of the Hénon map"""
    def __init__(self):
        pass

    def scan(self):
        pass

    def save_values(self, f, label="SixTrack LHC no bb flat"):
        self.label = label
        data_dict = {
            "label": label,
            "top": self.top,
            "steps": self.steps,
            "starting_radius": self.starting_radius,
            "times": self.times,
            "max_turns": self.max_turns
        }
        with open(f, 'wb') as destination:
            pickle.dump(data_dict, destination, protocol=4)

    @staticmethod
    def generate_instance(epsilon, top, steps, starting_radius=0.0001, cuda_device=None):
        """Create an uniform scan object

        Parameters
        ----------
        epsilon : float
            modulation intensity
        top : float
            maximum radius
        steps : int
            steps from zero to top (becomes steps * 2 + 1)
        starting_radius : float, optional
            from which position we have to start with the actual computation, by default 0.0001
        cuda_device : bool, optional
            do we have a CUDA capable device (make it manual), by default None

        Returns
        -------
        object
            uniform_scan object
        """        
        if cuda_device == None:
            cuda_device = cuda.is_available()
        if cuda_device:
            return gpu_uniform_scan(epsilon, top, steps, starting_radius)
        else:
            return cpu_uniform_scan(epsilon, top, steps, starting_radius)


class cpu_uniform_scan(uniform_scan):
    def __init__(self, epsilon, top, steps, starting_radius=0.0001):
        self.epsilon = epsilon
        self.top = top
        self.steps = steps
        self.starting_radius = starting_radius

        self.coords = np.linspace(-top, top, steps * 2 + 1)
        self.X, self.PX, self.Y, self.PY = np.meshgrid(
            self.coords, self.coords, self.coords, self.coords)

        self.X2 = np.power(self.X, 2)
        self.PX2 = np.power(self.PX, 2)
        self.Y2 = np.power(self.Y, 2)
        self.PY2 = np.power(self.PY, 2)

        self.bool_mask = (
            self.X2
            + self.PX2
            + self.Y2
            + self.PY2
            >= np.power(starting_radius, 2)
        )

        self.X_f = self.X.flatten()
        self.PX_f = self.PX.flatten()
        self.Y_f = self.Y.flatten()
        self.PY_f = self.PY.flatten()
        self.bool_mask_f = self.bool_mask.flatten()

        self.times = np.zeros_like(self.X)
        self.times_f = self.times.flatten()

        self.n_samples = np.count_nonzero(self.bool_mask_f)
    
    def scan(self, max_turns):
        """Execute a scanning of everything

        Parameters
        ----------
        max_turns : int
            turn limit

        Returns
        -------
        ndarray
            4d array with stable iterations inside
        """
        self.max_turns = max_turns

        omega_x, omega_y = modulation(self.epsilon, self.n_samples)

        # Filling
        start = time.time()

        self.times_f = cpu.henon_map_to_the_end(
            self.X_f, self.PX_f, self.Y_f, self.PY_f, 100.0, max_turns, omega_x, omega_y, self.bool_mask_f)

        print("Elapsed time for execution: {} s".format(time.time() - start))

        self.times = self.times_f.reshape(
            (self.steps * 2 + 1, self.steps * 2 + 1, self.steps * 2 + 1, self.steps * 2 + 1))

        return self.times


class gpu_uniform_scan(uniform_scan):
    def __init__(self, epsilon, top, steps, starting_radius=0.0001):
        self.epsilon = epsilon
        self.top = top
        self.steps = steps
        self.starting_radius = starting_radius

        self.coords = np.linspace(-top, top, steps * 2 + 1)
        self.X, self.PX, self.Y, self.PY = np.meshgrid(
            self.coords, self.coords, self.coords, self.coords,
            indexing='ij')

        self.X2 = np.power(self.X, 2)
        self.PX2 = np.power(self.PX, 2)
        self.Y2 = np.power(self.Y, 2)
        self.PY2 = np.power(self.PY, 2)

        self.bool_mask = (
            self.X2
            + self.PX2
            + self.Y2
            + self.PY2
            >= np.power(starting_radius, 2)
        )

        self.X_f = self.X.flatten()
        self.PX_f = self.PX.flatten()
        self.Y_f = self.Y.flatten()
        self.PY_f = self.PY.flatten()
        self.bool_mask_f = self.bool_mask.flatten()

        self.times = np.zeros_like(self.X)
        self.times_f = self.times.flatten()

        self.n_samples = np.count_nonzero(self.bool_mask_f)

    def scan(self, max_turns):
        """Execute a scanning of everything

        Parameters
        ----------
        max_turns : int
            turn limit

        Returns
        -------
        ndarray
            4d array with stable iterations inside
        """        
        threads_per_block = 512
        blocks_per_grid = 10

        d_x = cuda.to_device(self.X_f)
        d_px = cuda.to_device(self.PX_f)
        d_y = cuda.to_device(self.Y_f)
        d_py = cuda.to_device(self.PY_f)
        d_times = cuda.to_device(self.times_f)
        d_bool_mask = cuda.to_device(self.bool_mask_f)

        self.max_turns = max_turns

        omega_x, omega_y = modulation(self.epsilon, self.n_samples)

        d_omega_x = cuda.to_device(omega_x)
        d_omega_y = cuda.to_device(omega_y)

        # Filling
        start = time.time()

        gpu.henon_map_to_the_end[blocks_per_grid, threads_per_block](
            d_x, d_px, d_y, d_py, d_times, 100.0, max_turns, d_omega_x, d_omega_y, d_bool_mask
        )

        print("Elapsed time for execution: {} s".format(time.time() - start))

        d_times.copy_to_host(self.times_f)
        self.times = self.times_f.reshape(
            (self.steps * 2 + 1, self.steps * 2 + 1, self.steps * 2 + 1, self.steps * 2 + 1))

        return self.times


class radial_scan(object):
    """This class contains most of the tools required for doing a precise and on point radial scan for Dynamic Aperture estimations. It's a bit messy tho...
    """
    def __init__(self):
        pass

    def compute(self):
        pass

    def dummy_compute(self):
        pass

    def reset(self):
        pass

    def get_data(self):
        """Get the data
        
        Returns
        -------
        ndarray
            The data intended as last stable radius for the given amount of turns.
        """
        return np.transpose(np.asarray(self.container)) * self.dr

    @staticmethod
    def generate_instance(dr, alpha, theta1, theta2, epsilon, starting_position=0.0, cuda_device=None):
        """init an henon optimized radial tracker!
        
        Parameters
        ----------
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
        
        Returns
        -------
        Optimized instance
            optimized instance of the class (CPU or GPU)
        """
        if cuda_device == None:
            cuda_device = cuda.is_available()
        if cuda_device:
            return gpu_radial_scan(dr, alpha, theta1, theta2, epsilon, starting_position)
        else:
            return cpu_radial_scan(dr, alpha, theta1, theta2, epsilon, starting_position)

    def save_values(self, f, label="Hénon map scanning"):
        self.label = label
        data_dict = {
            "label": label,
            "alpha": self.alpha,
            "theta1": self.theta1,
            "theta2": self.theta2,
            "dr": self.dr,
            "starting_position": self.starting_position,
            "starting_step": 0, # this has its meaning in the bigger picture, trust me!
            "values": np.transpose(self.steps),
            "max_turns": self.sample_list[0],
            "min_turns": self.sample_list[-1]
        }
        with open(f, 'wb') as destination:
            pickle.dump(data_dict, destination, protocol=4)


class gpu_radial_scan(radial_scan):
    def __init__(self, dr, alpha, theta1, theta2, epsilon, starting_position=0.0):
        assert alpha.size == theta1.size
        assert alpha.size == theta2.size
        assert starting_position >= 0.0

        # save data as members
        self.dr = dr
        self.alpha = alpha
        self.theta1 = theta1
        self.theta2 = theta2
        self.epsilon = epsilon
        self.limit = 100.0
        self.starting_position = starting_position

        # prepare data
        self.starting_step = int(starting_position / dr)
        self.step = np.ones(alpha.shape, dtype=np.int) * int(starting_position / dr)

        # make container
        self.container = []

        # load vectors to gpu
        self.d_alpha = cuda.to_device(np.ascontiguousarray(self.alpha))
        self.d_theta1 = cuda.to_device(np.ascontiguousarray(self.theta1))
        self.d_theta2 = cuda.to_device(np.ascontiguousarray(self.theta2))
        self.d_step = cuda.to_device(np.ascontiguousarray(self.step))

    def reset(self):
        """Resets the engine.
        """
        self.container = []
        self.step = np.ones(self.alpha.shape, dtype=np.int) * \
            int(self.starting_position / self.dr)
        self.d_step = cuda.to_device(self.step)

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
        self.sample_list = sample_list
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

    def block_compute(self, max_turns, min_turns):
        """Optimize block computation for ending up with a proper steps block!

        Parameters
        ----------
        max_turns : int
            max number of turns
        min_turns : int
            min number of turns

        Returns
        -------
        ndarray
            the steps array
        """
        # precomputation
        self.compute([min_turns])

        # computation
        maximum = np.max(self.container)
        minimum = np.min(self.container)

        self.steps = np.zeros((self.alpha.shape[0], maximum))
        rs = (np.arange(maximum) + 2) * self.dr
        bool_mask = rs > (minimum * self.dr) / 2

        bb, aa = np.meshgrid(bool_mask, self.alpha, indexing='ij')
        rr, aa = np.meshgrid(rs, self.alpha, indexing='ij')
        rr, th1 = np.meshgrid(rs, self.theta1, indexing='ij')
        rr, th2 = np.meshgrid(rs, self.theta2, indexing='ij')

        bb = bb.flatten()
        aa = aa.flatten()
        th1 = th1.flatten()
        th2 = th2.flatten()
        rr = rr.flatten()

        x, px, y, py = polar_to_cartesian(rr, aa, th1, th2)
        steps = np.zeros_like(x, dtype=np.int)

        threads_per_block = 512
        blocks_per_grid = 10

        omega_x, omega_y = modulation(self.epsilon, max_turns)

        d_bb = cuda.to_device(bb)
        d_omega_x = cuda.to_device(omega_x)
        d_omega_y = cuda.to_device(omega_y)
        d_x = cuda.to_device(x)
        d_px = cuda.to_device(px)
        d_y = cuda.to_device(y)
        d_py = cuda.to_device(py)
        d_steps = cuda.to_device(steps)

        gpu.henon_map_to_the_end[blocks_per_grid, threads_per_block](
            d_x, d_px, d_y, d_py,
            d_steps, self.limit, max_turns,
            d_omega_x, d_omega_y,
            d_bb
        )

        d_steps.copy_to_host(steps)
        self.steps = steps.reshape(
            (rs.shape[0], self.alpha.shape[0]))

        return self.steps

class cpu_radial_scan(radial_scan):
    def __init__(self, dr, alpha, theta1, theta2, epsilon, starting_position=0.0):
        assert alpha.size == theta1.size
        assert alpha.size == theta2.size

        # save data as members
        self.dr = dr
        self.alpha = alpha
        self.theta1 = theta1
        self.theta2 = theta2
        self.epsilon = epsilon
        self.limit = 100.0
        self.starting_position = starting_position

        # prepare data
        self.step = np.ones(alpha.shape, dtype=np.int) * int(starting_position / dr)

        # make container
        self.container = []

    def reset(self):
        """Resets the engine.
        """
        self.container = []
        self.step = np.ones(self.alpha.shape, dtype=np.int) * \
            int(self.starting_position / self.dr)

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
        self.sample_list = sample_list
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

    def block_compute(self, max_turns, min_turns):
        """Optimize block computation for ending up with a proper steps block!

        Parameters
        ----------
        max_turns : int
            max number of turns
        min_turns : int
            min number of turns

        Returns
        -------
        ndarray
            the steps array
        """
        # precomputation
        self.compute([min_turns])

        # computation
        maximum = np.max(self.container)
        minimum = np.min(self.container)

        rs = (np.arange(maximum) + 2) * self.dr
        bool_mask = rs > (minimum * self.dr) / 2

        bb, aa = np.meshgrid(bool_mask, self.alpha, indexing='ij')
        rr, aa = np.meshgrid(rs, self.alpha, indexing='ij')
        rr, th1 = np.meshgrid(rs, self.theta1, indexing='ij')
        rr, th2 = np.meshgrid(rs, self.theta2, indexing='ij')

        bb = bb.flatten()
        aa = aa.flatten()
        th1 = th1.flatten()
        th2 = th2.flatten()
        rr = rr.flatten()

        x, px, y, py = polar_to_cartesian(rr, aa, th1, th2)
        
        omega_x, omega_y = modulation(self.epsilon, max_turns)

        steps = cpu.henon_map_to_the_end(
            x, px, y, py,
            self.limit, max_turns,
            omega_x, omega_y,
            bb
        )

        self.steps = steps.reshape(
            (rs.shape[0], self.alpha.shape[0]))

        return self.steps
        

class full_track(object):
    def __init__(self):
        pass

    def compute(self):
        pass

    def get_data(self):
        """Get the data
        
        Returns
        -------
        tuple of 2D ndarray [n_iterations, n_samples]
            (radius, alpha, theta1, theta2)
        """
        return cpu.cartesian_to_polar(self.x, self.px, self.y, self.py)

    def accumulate_and_return(self, n_sectors):
        """Returns the summed results (power 4 as documented)
        
        Parameters
        ----------
        n_sectors : int
            number of sectors to consider in the 2 theta space
        
        Returns
        -------
        ndarray
            list of values for the different istances considered.
        """        
        radius, alpha, th1, th2 = cpu.cartesian_to_polar(
            self.x, self.px, self.y, self.py)
        
        self.count_matrix, self.matrices, result = cpu.accumulate_and_return(radius, alpha, th1, th2, n_sectors)
        
        return result
    
    def recursive_accumulation(self):
        """Executes a recursive accumulation in order to test lower binning values.
        N.B. execute "accumulate_and_return first!!!"

        Returns
        -------
        tuple of lists
            Tuple of lists with (count_matrices, averages, results)
        """
        return cpu.recursive_accumulation(self.count_matrix, self.matrices)        

    @staticmethod
    def generate_instance(radius, alpha, theta1, theta2, iters, epsilon, cuda_device=None):
        """Generate an instance of the class
        
        Parameters
        ----------
        radius : ndarray
            radius to consider
        alpha : ndarray
            initial angle
        theta1 : ndarray
            initial theta1
        theta2 : ndarray
            initial theta2
        iters : ndarray
            n_iterations to perform
        epsilon : float
            intensity of the modulation
        
        Returns
        -------
        class instance
            optimized class instance
        """        
        if cuda_device == None:
            cuda_device = cuda.is_available()
        if cuda_device:
            return gpu_full_track(radius, alpha, theta1, theta2, iters, epsilon)
        else:
            return cpu_full_track(radius, alpha, theta1, theta2, iters, epsilon)


class gpu_full_track(full_track):
    def __init__(self, radius, alpha, theta1, theta2, iters, epsilon):
        assert alpha.size == theta1.size
        assert alpha.size == theta2.size
        assert alpha.size == radius.size

        # save data as members
        self.radius = radius
        self.alpha = alpha
        self.theta1 = theta1
        self.theta2 = theta2
        self.epsilon = epsilon
        self.iters = iters

        self.max_iters = np.max(self.iters)

        # make containers
        self.x = np.empty((self.max_iters, alpha.size)) * np.nan
        self.px = np.empty((self.max_iters, alpha.size)) * np.nan
        self.y = np.empty((self.max_iters, alpha.size)) * np.nan
        self.py = np.empty((self.max_iters, alpha.size)) * np.nan

        self.x[0, :], self.px[0, :], self.y[0, :], self.py[0, :] = gpu.actual_polar_to_cartesian(radius, alpha, theta1, theta2)
    
    def compute(self):
        """Compute the tracking
        
        Returns
        -------
        tuple of 2D ndarray [n_iterations, n_samples]
            (radius, alpha, theta1, theta2)
        """

        # load vectors to gpu

        d_x = cuda.to_device(self.x)
        d_px = cuda.to_device(self.px)
        d_y = cuda.to_device(self.y)
        d_py = cuda.to_device(self.py)
        d_iters = cuda.to_device(self.iters)

        threads_per_block = 512
        blocks_per_grid = 10

        omega_x, omega_y = modulation(self.epsilon, self.max_iters)

        d_omega_x = cuda.to_device(omega_x)
        d_omega_y = cuda.to_device(omega_y)

        # Execution
        gpu.henon_full_track[blocks_per_grid, threads_per_block](
            d_x, d_px, d_y, d_py,
            d_iters, d_omega_x, d_omega_y
        )

        d_x.copy_to_host(self.x)
        d_y.copy_to_host(self.y)
        d_px.copy_to_host(self.px)
        d_py.copy_to_host(self.py)
        d_iters.copy_to_host(self.iters)

        return self.x, self.px, self.y, self.py


class cpu_full_track(full_track):
    def __init__(self, radius, alpha, theta1, theta2, iters, epsilon):
        assert alpha.size == theta1.size
        assert alpha.size == theta2.size
        assert alpha.size == radius.size

        # save data as members
        self.radius = radius
        self.alpha = alpha
        self.theta1 = theta1
        self.theta2 = theta2
        self.epsilon = epsilon
        self.iters = iters
        self.max_iters = np.max(self.iters)

        # make containers
        self.x = np.zeros((self.max_iters, alpha.size)) * np.nan
        self.px = np.zeros((self.max_iters, alpha.size)) * np.nan
        self.y = np.zeros((self.max_iters, alpha.size)) * np.nan
        self.py = np.zeros((self.max_iters, alpha.size)) * np.nan

        self.x[0, :], self.px[0, :], self.y[0, :], self.py[0, :] = cpu.polar_to_cartesian(radius, alpha, theta1, theta2)

    def compute(self):
        """Compute the tracking
        
        Returns
        -------
        tuple of 2D ndarray [n_iterations, n_samples]
            (radius, alpha, theta1, theta2)
        """
        omega_x, omega_y = modulation(self.epsilon, self.max_iters)
        # Execution
        self.x, self.px, self.y, self.py = cpu.henon_full_track(
            self.x, self.px, self.y, self.py,
            self.iters, omega_x, omega_y
        )
        return self.x, self.px, self.y, self.py
