import matplotlib.pyplot as plt
from numba import cuda, jit, njit, prange
import numpy as np
from tqdm import tqdm
import scipy.integrate as integrate
import pickle
import time
import tempfile
import h5py

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

    def save_values(self, f, label="Henon map"):
        self.db.flush()
        dest = h5py.File(f, mode="w")
        
        dest.attrs["label"] = label
        dest.attrs["epsilon"] = self.db.attrs["epsilon"]
        dest.attrs["top"] = self.db.attrs["top"]
        dest.attrs["steps"] = self.db.attrs["steps"]
        dest.attrs["starting_radius"] = self.db.attrs["starting_radius"]
        dest.attrs["coordinates"] = self.db.attrs["coordinates"]
        dest.attrs["samples"] = self.db.attrs["samples"]
        dest.attrs["max_turns"] = self.db.attrs["max_turns"]

        g = dest.create_group("data")
        self.db.copy("data/times", g)

        dest.create_dataset(
            "/data/weights", (self.samples, self.samples, self.samples, self.samples), dtype=np.float, compression="lzf")
        dest.close()

    @staticmethod
    def generate_instance(epsilon, top, steps, starting_radius=0.0001, cuda_device=None, tempdir=None):
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
            return gpu_uniform_scan(epsilon, top, steps, starting_radius, tempdir)
        else:
            return cpu_uniform_scan(epsilon, top, steps, starting_radius, tempdir)


class cpu_uniform_scan(uniform_scan):
    def __init__(self, epsilon, top, steps, starting_radius=0.0001, tempdir=None):
        self.tf = tempfile.TemporaryFile(dir=tempdir)
        self.db = h5py.File(self.tf, mode="w")

        self.samples = steps * 2 + 1
        self.coords = np.linspace(-top, top, self.samples)
        
        self.db.attrs["epsilon"] = epsilon
        self.db.attrs["top"] = top
        self.db.attrs["steps"] = steps
        self.db.attrs["starting_radius"] = starting_radius
        self.db.attrs["samples"] = self.samples
        self.db.attrs["coordinates"] = self.coords

        self.bool_mask = self.db.create_dataset(
            "/data/bool_mask", (self.samples, self.samples, self.samples, self.samples), dtype=np.bool, compression="lzf")

        self.coords2 = np.power(self.coords, 2)
        for i in tqdm(range(len(self.coords)), desc="Make the boolean mask"):
            px, y, py = np.meshgrid(self.coords2, self.coords2, self.coords2)
            self.bool_mask[i] = (
                self.coords[i] ** 2 
                + px
                + y
                + py
                >= starting_radius ** 2
            )

        self.times = self.db.create_dataset(
            "/data/times", (self.samples, self.samples, self.samples, self.samples), dtype=np.int32, compression="lzf")
        
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
        self.db.attrs["max_turns"] = max_turns

        omega_x, omega_y = modulation(self.db.attrs["epsilon"], max_turns)

        for i in tqdm(range(len(self.times))):
            px, y, py = np.meshgrid(self.coords, self.coords, self.coords)
            x = np.ones_like(px) * self.coords[i]
            self.times[i] = cpu.henon_map_to_the_end(
                x, px, y, py, 10.0, max_turns, omega_x, omega_y, self.bool_mask[i]
            )

    def scan_octo(self, max_turns, mu):
        """Execute a scanning of everything

        Parameters
        ----------
        max_turns : int
            turn limit

        mu : float
            mu parameter

        Returns
        -------
        ndarray
            4d array with stable iterations inside
        """
        self.db.attrs["max_turns"] = max_turns
        self.db.attrs["mu"] = mu

        omega_x, omega_y = modulation(self.db.attrs["epsilon"], max_turns)

        for i in tqdm(range(len(self.times))):
            px, y, py = np.meshgrid(self.coords, self.coords, self.coords)
            x = np.ones_like(px) * self.coords[i]
            self.times[i] = cpu.octo_henon_map_to_the_end(
                x, px, y, py, 10.0, max_turns, omega_x, omega_y, mu, self.bool_mask[i]
            )


class gpu_uniform_scan(uniform_scan):
    def __init__(self, epsilon, top, steps, starting_radius=0.0001, tempdir=None):
        self.tf = tempfile.TemporaryFile(dir=tempdir)
        self.db = h5py.File(self.tf, mode="w")

        self.samples = steps * 2 + 1
        self.coords = np.linspace(-top, top, self.samples)

        self.db.attrs["epsilon"] = epsilon
        self.db.attrs["top"] = top
        self.db.attrs["steps"] = steps
        self.db.attrs["starting_radius"] = starting_radius
        self.db.attrs["samples"] = self.samples
        self.db.attrs["coordinates"] = self.coords

        self.bool_mask = self.db.create_dataset(
            "/data/bool_mask", (self.samples, self.samples, self.samples, self.samples), dtype=np.bool, compression="lzf")

        self.coords2 = np.power(self.coords, 2)
        for i in tqdm(range(len(self.coords)), desc="Make the boolean mask"):
            px, y, py = np.meshgrid(self.coords2, self.coords2, self.coords2)
            self.bool_mask[i] = (
                self.coords[i] ** 2
                + px
                + y
                + py
                >= starting_radius ** 2
            )

        self.times = self.db.create_dataset(
            "/data/times", (self.samples, self.samples, self.samples, self.samples), dtype=np.int32, compression="lzf")

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

        self.db.attrs["max_turns"] = max_turns

        omega_x, omega_y = modulation(self.db.attrs["epsilon"], max_turns)
        d_omega_x = cuda.to_device(omega_x)
        d_omega_y = cuda.to_device(omega_y)

        t_f = np.empty(shape=(self.samples, self.samples, self.samples), dtype=np.int32).flatten()

        for i in tqdm(range(len(self.times)), smoothing=1.0):
            px, y, py = np.meshgrid(self.coords, self.coords, self.coords)
            x = np.ones_like(px) * self.coords[i]

            d_x = cuda.to_device(x.flatten())
            d_px = cuda.to_device(px.flatten())
            d_y = cuda.to_device(y.flatten())
            d_py = cuda.to_device(py.flatten())
            d_times = cuda.to_device(np.zeros(x.size, dtype=np.int32))
            d_bool_mask = cuda.to_device(np.asarray(self.bool_mask[i]).flatten())

            gpu.henon_map_to_the_end[blocks_per_grid, threads_per_block](
                d_x, d_px, d_y, d_py, d_times, 10.0, max_turns, d_omega_x, d_omega_y, d_bool_mask
            )

            d_times.copy_to_host(t_f)
            self.times[i] = t_f.reshape(x.shape)
    
    def scan_octo(self, max_turns, mu):
        """Execute a scanning of everything

        Parameters
        ----------
        max_turns : int
            turn limit

        mu : float
            param

        Returns
        -------
        ndarray
            4d array with stable iterations inside
        """
        threads_per_block = 512
        blocks_per_grid = 10

        self.db.attrs["max_turns"] = max_turns
        self.db.attrs["mu"] = mu

        omega_x, omega_y = modulation(self.db.attrs["epsilon"], max_turns)
        d_omega_x = cuda.to_device(np.asarray(omega_x, dtype=np.float64))
        d_omega_y = cuda.to_device(np.asarray(omega_y, dtype=np.float64))

        t_f = np.empty(shape=(self.samples, self.samples,
                              self.samples), dtype=np.int32).flatten()

        for i in tqdm(range(len(self.times)), smoothing=1.0):
            px, y, py = np.meshgrid(self.coords, self.coords, self.coords)
            x = np.ones_like(px) * self.coords[i]

            d_x = cuda.to_device(np.asarray(x, dtype=np.float64).flatten())
            d_px = cuda.to_device(np.asarray(px, dtype=np.float64).flatten())
            d_y = cuda.to_device(np.asarray(y, dtype=np.float64).flatten())
            d_py = cuda.to_device(np.asarray(py, dtype=np.float64).flatten())
            d_times = cuda.to_device(np.zeros(x.size, dtype=np.int32))
            d_bool_mask = cuda.to_device(
                np.asarray(self.bool_mask[i]).flatten())

            gpu.octo_henon_map_to_the_end[blocks_per_grid, threads_per_block](
                d_x, d_px, d_y, d_py, d_times, 10.0, max_turns, d_omega_x, d_omega_y, np.float64(mu), d_bool_mask
            )

            d_times.copy_to_host(t_f)
            self.times[i] = t_f.reshape(x.shape)


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
        

class radial_block(object):
    def __init__(self):
        pass

    def scan(self):
        pass

    def save_values(self, f, label="Henon map"):
        self.db.flush()
        dest = h5py.File(f, mode="w")

        dest.attrs["label"] = label
        dest.attrs["epsilon"] = self.db.attrs["epsilon"]
        dest.attrs["radial_samples"] = self.db.attrs["radial_samples"]
        dest.attrs["max_radius"] = self.db.attrs["max_radius"]
        dest.attrs["alpha"] = self.db.attrs["alpha"]
        dest.attrs["theta1"] = self.db.attrs["theta1"]
        dest.attrs["theta2"] = self.db.attrs["theta2"]
        dest.attrs["dr"] = self.db.attrs["dr"]
        dest.attrs["max_turns"] = self.db.attrs["max_turns"]
        
        g = dest.create_group("data")
        self.db.copy("data/times", g)

        dest.create_dataset(
            "/data/weights", (self.db.attrs["radial_samples"], len(self.db.attrs["alpha"]), len(self.db.attrs["theta1"]), len(self.db.attrs["theta2"])), dtype=np.float, compression="lzf")
        dest.close()
    
    @staticmethod
    def generate_instance(radial_samples, alpha, theta1, theta2, epsilon, max_radius=1.0, starting_radius=0.0, cuda_device=None, tempdir=None):
        if cuda_device == None:
            cuda_device = cuda.is_available()
        if cuda_device:
            return gpu_radial_block(radial_samples, alpha, theta1, theta2, epsilon, max_radius, starting_radius, tempdir)
        else:
            return cpu_radial_block(radial_samples, alpha, theta1, theta2, epsilon, max_radius, starting_radius, tempdir)

        
class cpu_radial_block(radial_block):
    def __init__(self, radial_samples, alpha, theta1, theta2, epsilon, max_radius=1.0, starting_radius=0.0, tempdir=None):
        assert alpha.size == theta1.size
        assert alpha.size == theta2.size
        assert max_radius > 0.0
        
        self.tf = tempfile.TemporaryFile(dir=tempdir)
        self.db = h5py.File(self.tf, mode="w")

        self.db.attrs["epsilon"] = epsilon
        self.db.attrs["radial_samples"] = radial_samples
        self.db.attrs["starting_radius"] = starting_radius
        self.starting_radius = starting_radius
        self.db.attrs["max_radius"] = max_radius
        self.alpha = alpha
        self.theta1 = theta1
        self.theta2 = theta2
        self.db.attrs["alpha"] = alpha
        self.db.attrs["theta1"] = theta1
        self.db.attrs["theta2"] = theta2

        self.r_list, self.dr = np.linspace(0, max_radius, radial_samples + 1, retstep=True)
        self.r_list = self.r_list[1:]
        self.db.attrs["dr"] = self.dr

        self.bool_mask = self.db.create_dataset(
            "/data/bool_mask", (radial_samples, len(alpha), len(theta1), len(theta2)), dtype=np.bool, compression="lzf")
        
        for i in tqdm(range(radial_samples)):
            self.bool_mask[i] = self.r_list[i] >= starting_radius

        self.times = self.db.create_dataset(
            "/data/times", (radial_samples, len(alpha), len(theta1), len(theta2)), dtype=np.int32, compression="lzf")
    
    def scan(self, max_turns):
        self.db.attrs["max_turns"] = max_turns

        omega_x, omega_y = modulation(self.db.attrs["epsilon"], max_turns)

        aa, th1, th2 = np.meshgrid(
            self.alpha, self.theta1, self.theta2, indexing='ij'
        )

        for i in tqdm(range(len(self.times)), smoothing=1.0):
            if self.r_list[i] < self.starting_radius:
                self.times[i] = max_turns
            else:
                x, px, y, py = polar_to_cartesian(self.r_list[i], aa, th1, th2)

                self.times[i] = cpu.henon_map_to_the_end(
                    x, px, y, py, 10.0, max_turns, omega_x, omega_y, self.bool_mask[i]
                )
    
    def scan_octo(self, max_turns, mu):
        self.db.attrs["max_turns"] = max_turns
        self.db.attrs["mu"] = mu

        omega_x, omega_y = modulation(self.db.attrs["epsilon"], max_turns)

        aa, th1, th2 = np.meshgrid(
            self.alpha, self.theta1, self.theta2, indexing='ij'
        )

        for i in tqdm(range(len(self.times)), smoothing=1.0):
            if self.r_list[i] < self.starting_radius:
                self.times[i] = max_turns
            else:
                x, px, y, py = polar_to_cartesian(self.r_list[i], aa, th1, th2)

                self.times[i] = cpu.octo_henon_map_to_the_end(
                    x, px, y, py, 10.0, max_turns, omega_x, omega_y, mu, self.bool_mask[i]
                )
        

class gpu_radial_block(radial_block):
    def __init__(self, radial_samples, alpha, theta1, theta2, epsilon, max_radius=1.0, starting_radius=0.0, tempdir=None):
        assert alpha.size == theta1.size
        assert alpha.size == theta2.size
        assert max_radius > 0.0

        self.tf = tempfile.TemporaryFile(dir=tempdir)
        self.db = h5py.File(self.tf, mode="w")

        self.db.attrs["epsilon"] = epsilon
        self.db.attrs["radial_samples"] = radial_samples
        self.db.attrs["starting_radius"] = starting_radius
        self.starting_radius = starting_radius
        self.db.attrs["max_radius"] = max_radius
        self.alpha = alpha
        self.theta1 = theta1
        self.theta2 = theta2
        self.db.attrs["alpha"] = alpha
        self.db.attrs["theta1"] = theta1
        self.db.attrs["theta2"] = theta2

        self.r_list, self.dr = np.linspace(
            0, max_radius, radial_samples + 1, retstep=True)
        self.r_list = self.r_list[1:]
        self.db.attrs["dr"] = self.dr

        self.bool_mask = self.db.create_dataset(
            "/data/bool_mask", (radial_samples, len(alpha), len(theta1), len(theta2)), dtype=np.bool, compression="lzf")

        for i in tqdm(range(radial_samples)):
            self.bool_mask[i] = self.r_list[i] >= starting_radius

        self.times = self.db.create_dataset(
            "/data/times", (radial_samples, len(alpha), len(theta1), len(theta2)), dtype=np.int32, compression="lzf")

    def scan(self, max_turns):
        threads_per_block = 512
        blocks_per_grid = 10

        self.db.attrs["max_turns"] = max_turns

        omega_x, omega_y = modulation(self.db.attrs["epsilon"], max_turns)
        d_omega_x = cuda.to_device(omega_x)
        d_omega_y = cuda.to_device(omega_y)

        t_f = np.empty(shape=(len(self.alpha), len(
            self.theta1), len(self.theta2)), dtype=np.int32).flatten()
        aa, th1, th2 = np.meshgrid(
            self.alpha, self.theta1, self.theta2, indexing='ij'
        )

        for i in tqdm(range(len(self.times)), smoothing=1.0):
            if self.r_list[i] < self.starting_radius:
                self.times[i] = max_turns
            else:
                x, px, y, py = polar_to_cartesian(self.r_list[i], aa, th1, th2)
                d_x = cuda.to_device(x.flatten())
                d_px = cuda.to_device(px.flatten())
                d_y = cuda.to_device(y.flatten())
                d_py = cuda.to_device(py.flatten())
                d_bool_mask = cuda.to_device(
                    np.asarray(self.bool_mask[i]).flatten())
                d_times = cuda.to_device(np.zeros(x.size, dtype=np.int32))

                gpu.henon_map_to_the_end[blocks_per_grid, threads_per_block](
                    d_x, d_px, d_y, d_py, d_times, 10.0, max_turns, d_omega_x, d_omega_y, d_bool_mask
                )

                d_times.copy_to_host(t_f)
                self.times[i] = t_f.reshape(x.shape)
    
    def scan_octo(self, max_turns, mu):
        threads_per_block = 512
        blocks_per_grid = 10

        self.db.attrs["max_turns"] = max_turns
        self.db.attrs["mu"] = mu

        omega_x, omega_y = modulation(self.db.attrs["epsilon"], max_turns)
        d_omega_x = cuda.to_device(np.asarray(omega_x, dtype=np.float64))
        d_omega_y = cuda.to_device(np.asarray(omega_y, dtype=np.float64))

        t_f = np.empty(shape=(len(self.alpha), len(
            self.theta1), len(self.theta2)), dtype=np.int32).flatten()
        aa, th1, th2 = np.meshgrid(
            self.alpha, self.theta1, self.theta2, indexing='ij'
        )

        for i in tqdm(range(len(self.times)), smoothing=1.0):
            if self.r_list[i] < self.starting_radius:
                self.times[i] = max_turns
            else:
                x, px, y, py = polar_to_cartesian(self.r_list[i], aa, th1, th2)
                d_x = cuda.to_device(np.asarray(x, dtype=np.float64).flatten())
                d_px = cuda.to_device(np.asarray(px, dtype=np.float64).flatten())
                d_y = cuda.to_device(np.asarray(y, dtype=np.float64).flatten())
                d_py = cuda.to_device(np.asarray(py, dtype=np.float64).flatten())
                d_bool_mask = cuda.to_device(np.asarray(self.bool_mask[i]).flatten())
                d_times = cuda.to_device(np.zeros(x.size, dtype=np.int32))

                gpu.octo_henon_map_to_the_end[blocks_per_grid, threads_per_block](
                    d_x, d_px, d_y, d_py, d_times, 10.0, max_turns, d_omega_x, d_omega_y, np.float64(mu), d_bool_mask
                )

                d_times.copy_to_host(t_f)
                self.times[i] = t_f.reshape(x.shape)


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


class uniform_analyzer(object):
    def __init__(self, hdf5_dir):
        self.db = h5py.File(hdf5_dir, mode="r+")
        self.samples = self.db.attrs["samples"]
        self.coords = self.db.attrs["coordinates"]
        self.coords2 = np.power(self.coords, 2)
        self.times = self.db["/data/times"]
        self.weights = self.db["/data/weights"]

    def assign_weights(self, f=lambda x, px, y, py: 1.0, radial_cut=-1.0):
        if radial_cut == -1.0:
            radial_cut = self.db.attrs["top"]

        for i in tqdm(range(self.samples), desc="assigning weights"):
            px, y, py = np.meshgrid(self.coords, self.coords, self.coords)
            self.weights[i] = f(self.coords[i], px, y, py) * (
                np.power(self.coords[i], 2) + np.power(px, 2) + np.power(y, 2) + np.power(py, 2) <= np.power(radial_cut, 2)
            )

    def compute_loss(self, sample_list, normalization=True):
        prelim_values = np.empty(self.samples)        
        for i in tqdm(range(self.samples), desc="baseline integration"):
            prelim_values[i] = integrate.trapz(
                integrate.trapz(
                    integrate.trapz(
                        self.weights[i],
                        x=self.coords
                    ),
                    x=self.coords
                ),
                x=self.coords
            )
        baseline = integrate.trapz(prelim_values, x=self.coords)

        values = np.empty(len(sample_list))
        for j, sample in tqdm(enumerate(sample_list), desc="other integrals...", total=len(sample_list)):
            prelim_values = np.empty(self.samples)
            for i in range(self.samples):
                prelim_values[i] = integrate.trapz(
                    integrate.trapz(
                        integrate.trapz(
                            self.weights[i] * (self.times[i] >= sample),
                            x=self.coords
                        ),
                        x=self.coords
                    ),
                    x=self.coords
                )
            values[j] = integrate.trapz(prelim_values, x=self.coords)

        if normalization:
            values /= baseline

        return values

    def compute_loss_cut(self, cut):
        prelim_values = np.empty(self.samples)
        for i in tqdm(range(self.samples), desc="baseline integration"):
            px, y, py = np.meshgrid(self.coords2, self.coords2, self.coords2)
            prelim_values[i] = integrate.trapz(
                integrate.trapz(
                    integrate.trapz(
                        self.weights[i] * (np.power(self.coords[i], 2) + px + y + py <= np.power(cut, 2)),
                        x=self.coords
                    ),
                    x=self.coords
                ),
                x=self.coords
            )
        return integrate.trapz(prelim_values, self.coords)


class uniform_radial_scanner(object):
    """This class is for analyzing the loss values of a (somewhat) angular uniform scan"""

    def __init__(self, hdf5_dir):
        self.db = h5py.File(hdf5_dir, mode="r+")
        self.max_radius = self.db.attrs["max_radius"]
        self.samples = self.db.attrs["radial_samples"]
        self.r_list, self.dr = np.linspace(
            0, self.max_radius, self.samples, retstep=True)
        self.r_list = self.r_list[1:]
        self.alpha = self.db.attrs["alpha"]
        self.theta1 = self.db.attrs["theta1"]
        self.theta2 = self.db.attrs["theta2"]
        self.dr = self.db.attrs["dr"]

        self.times = self.db["/data/times"]
        self.weights = self.db["/data/weights"]

    @staticmethod
    def static_extract_radiuses(n_alpha, n_theta1, n_theta2, samples, times, dr, radius_db):
        for index, i1 in enumerate(range(n_alpha)):
            #print(index, "/", n_alpha, flush=True)
            #for i2 in range(n_theta1):
                #for i3 in range(n_theta2):
            temp = times[:, i1, :, :]
            values = np.empty((len(samples), n_theta1, n_theta2))
            for i, sample in enumerate(samples):
                values[i] = np.argmin(temp >= sample, axis=0) - 1
            values[values < 0] = len(values)
            values = (values + 1) * dr
            radius_db[:, i1, :, :] = values

    def compute_DA_standard(self, sample_list, get_radiuses=False):
        self.sample_list = sample_list
        """
        try:
            self.db_sample_list = self.db.require_dataset(
                "/data/DA_samples", shape=(len(sample_list),), dtype=np.int32, exact=True)
        except TypeError:
            del self.db["/data/DA_samples"]
            self.db_sample_list = self.db.require_dataset(
                "/data/DA_samples", shape=(len(sample_list),), dtype=np.int32, exact=True)
        
        try:
            self.radiuses = self.db.require_dataset(
                "/data/DA_radiuses", shape=(len(sample_list), len(self.alpha), len(self.theta1), len(self.theta2)), dtype=np.float, exact=True)
        except TypeError:
            del self.db["/data/DA_radiuses"]
            self.radiuses = self.db.require_dataset(
                "/data/DA_radiuses", shape=(len(sample_list), len(self.alpha), len(self.theta1), len(self.theta2)), dtype=np.float, exact=True)
        """
        radiuses = np.empty((len(sample_list), len(self.alpha), len(self.theta1,), len(self.theta1)))

        self.static_extract_radiuses(
            len(self.alpha), len(self.theta1), len(self.theta2),
            sample_list, self.times, self.dr, radiuses)

        mod_radiuses = np.power(radiuses, 4)
        mod_radiuses = integrate.trapz(mod_radiuses, x=self.theta2)
        mod_radiuses = integrate.trapz(mod_radiuses, x=self.theta1)
        mod_radiuses = integrate.trapz(mod_radiuses, x=self.alpha)

        self.DA = np.power(
            mod_radiuses / (2 * self.theta1[-1] * self.theta2[-1]), 1/4)
        
        e_alpha = np.mean(
            np.absolute(radiuses[:, 1:] - radiuses[:, :-1]),
            axis=(1, 2, 3)) ** 2
        e_theta1 = np.mean(
            np.absolute(radiuses[:, :, 1:] - radiuses[:, :, :-1]),
            axis=(1, 2, 3)) ** 2
        e_theta2 = np.mean(
            np.absolute(radiuses[:, :, :, 1:] - radiuses[:, :, :, :-1]),
            axis=(1, 2, 3)) ** 2
        e_radius = self.dr ** 2
        self.error = np.sqrt(
            (e_radius + e_alpha + e_theta1 + e_theta2) / 4)
        
        if not get_radiuses:
            return self.DA, self.error
        else:
            return self.DA, self.error, radiuses

    def create_weights_in_dataset(self, file_destination, f=lambda r, a, th1, th2: np.ones_like(a)):
        dest = h5py.File(file_destination, mode="w")
        weights = dest.create_dataset(
            "/data/weights", (self.db.attrs["radial_samples"], len(self.db.attrs["alpha"]), len(self.db.attrs["theta1"]), len(self.db.attrs["theta2"])), dtype=np.float, compression="lzf")
        
        for i in tqdm(range(0, len(self.r_list), 10), desc="assigning weights"):
            rr, aa, th1, th2 = np.meshgrid(
                self.r_list[i: min(i + 10, len(self.r_list))
                       ], self.alpha, self.theta1, self.theta2, indexing='ij'
            )
            weights[i: min(i + 10, len(self.r_list))
                         ] = f(rr, aa, th1, th2)
        
        dest.close()

    def assign_weights_from_file(self, file):
        self.weight_db = h5py.File(file, mode="r")
        self.weights = self.weight_db["/data/weights"]

    def assign_weights(self, f=lambda r, a, th1, th2: np.ones_like(a)):
        """Assign weights to the various radial samples computed (not-so-intuitive to setup, beware...).

        Parameters
        ----------
        f : lambda, optional
            the lambda to assign the weights with, by default returns r
            this lambda has to take as arguments
            r : float
                the radius
            a : float
                the alpha angle
            th1 : float
                the theta1 angle
            th2 : float
                the theta2 angle
        """
        aa, th1, th2 = np.meshgrid(
            self.alpha, self.theta1, self.theta2, indexing='ij'
        )
        for i, r in tqdm(enumerate(self.r_list), desc="assigning weights", total=len(self.r_list)):
            self.weights[i] = f(r, aa, th1, th2) 

    def compute_loss(self, sample_list, cutting_point=-1.0, normalization=True):
        """Compute the loss based on a boolean masking of the various timing values.

        Parameters
        ----------
        sample_list : ndarray
            list of times to use as samples
        cutting_point : float, optional
            radius to set-up as cutting point for normalization purposes, by default -1.0
        normalization : boolean, optional
            execute normalization? By default True

        Returns
        -------
        ndarray
            the values list measured (last element is the cutting point value 1.0 used for renormalization of the other results.)
        """
        if cutting_point == -1.0:
            cutting_point = self.db.attrs["max_radius"]

        prelim_values = np.empty(self.r_list.size)

        prelim_values[self.r_list > cutting_point] = 0.0
        
        for i in range(0, np.argmin(self.r_list <= cutting_point), 500):
            top_i = min(i + 500, np.argmin(self.r_list <= cutting_point))
            
            prelim_values[i : top_i] = integrate.trapz(
                integrate.trapz(
                    integrate.trapz(
                        self.weights[i : top_i],
                        self.theta2
                    ),
                    self.theta1
                ) * np.sin(self.alpha) * np.cos(self.alpha),
                self.alpha
            )
        baseline = integrate.trapz(prelim_values * np.power(self.r_list, 3), self.r_list)

        values = np.empty(len(sample_list))
        for j, sample in enumerate(sample_list):
            #print("integral", j, "/", len(sample_list), flush=True)
            prelim_values = np.empty(self.r_list.size)
            prelim_values[self.r_list > cutting_point] = 0.0

            for i in range(0, np.argmin(self.r_list <= cutting_point), 1000):
                top_i = min(i + 1000, np.argmin(self.r_list <= cutting_point))

                prelim_values[i: top_i] = integrate.trapz(
                    integrate.trapz(
                        integrate.trapz(
                            self.weights[i: top_i] * (self.times[i: top_i] >= sample),
                            self.theta2
                        ),
                        self.theta1
                    ) * np.sin(self.alpha) * np.cos(self.alpha),
                    self.alpha
                )
            values[j] = integrate.trapz(prelim_values * np.power(self.r_list, 3), self.r_list)

        if normalization:
            values /= baseline
        return np.abs(values)

    def compute_loss_cut(self, cutting_point=-1.0):
        """Compute the loss based on a simple DA cut.

        Parameters
        ----------
        cutting_point : float
            radius to set-up as cutting point

        Returns
        -------
        float
            the (not-normalized) value
        """
        prelim_values = np.empty(self.r_list.size)

        for i, r in tqdm(enumerate(self.r_list), desc="integration...", total=len(self.r_list)):
            if r > cutting_point:
                prelim_values[i] = 0.0
            else:
                prelim_values[i] = integrate.trapz(
                    integrate.trapz(
                        integrate.trapz(
                            self.weights[i],
                            self.theta2
                        ),
                        self.theta1
                    ) * np.sin(self.alpha) * np.cos(self.alpha),
                    self.alpha
                )
        return integrate.trapz(prelim_values * np.power(self.r_list, 3), self.r_list)


def assign_symmetric_gaussian(sigma=1.0, polar=True):
    if polar:
        def f(r, a, th1, th2):
            return (
                np.exp(- 0.5 * np.power(r / sigma, 2))
            )
    else:
        def f(x, px, y, py):
            return(
                np.exp(-0.5 * (np.power(x / sigma, 2.0) + np.power(y / sigma,
                                                                   2.0) + np.power(py / sigma, 2.0) + np.power(px / sigma, 2.0)))
            )
    return f


def assign_uniform_distribution(polar=True):
    if polar:
        def f(r, a, th1, th2):
            return (
                np.ones_like(r)
            )
    else:
        def f(x, px, y, py):
            return (
                np.ones_like(x)
            )
    return f


def assign_generic_gaussian(sigma_x, sigma_px, sigma_y, sigma_py, polar=True):
    if polar:
        def f(r, a, th1, th2):
            x, px, y, py = polar_to_cartesian(r, a, th1, th2)
            x /= sigma_x
            px /= sigma_px
            y /= sigma_y
            py /= sigma_py
            r, a, th1, th2 = cartesian_to_polar(x, px, y, py)
            return (
                np.exp(- np.power(r, 2) * 0.5) / (np.power(2 * np.pi, 2))
            )
    else:
        assert False  # Needs to be implemented lol
    return f
