import matplotlib.pyplot as plt
from numba import cuda, jit, njit, prange
import numpy as np
from tqdm import tqdm
import scipy.integrate as integrate
import pickle
import time
import tempfile
import h5py
from pynverse import inversefunc

from . import gpu_henon_core as gpu
from . import cpu_henon_core as cpu

from .cpu_henon_core import recursive_accumulation as cpu_accumulate_and_return

def polar_to_cartesian(radius, alpha, theta1, theta2):
    return cpu.polar_to_cartesian(radius, alpha, theta1, theta2)


def cartesian_to_polar(x, px, y, py):
    return cpu.cartesian_to_polar(x, px, y, py)


def circ_trapz(y, x, addendum):
    shape = list(y.shape)
    shape[-1] += 1
    new_y = np.empty(shape)
    new_y[..., :-1] = y[...]
    new_y[..., -1] = y[..., 0]

    new_x = np.empty(x.size + 1)
    new_x[:-1] = x[...]
    new_x[-1] = addendum

    return integrate.trapz(new_y, new_x)


@njit
def modulation(epsilon, n_elements, first_index=0, reversed=False):
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

    if not reversed:
        number_list = list(range(first_index, first_index + n_elements))
    else:
        number_list = list(range(first_index - n_elements, first_index))[::-1]

    omega_sum = np.array([
        np.sum(coefficients * np.cos(modulations * k)) for k in number_list
    ])
    omega_x = 0.168 * 2 * np.pi * (1 + epsilon * omega_sum)
    omega_y = 0.201 * 2 * np.pi * (1 + epsilon * omega_sum)
    return omega_x, omega_y


class partial_track(object):
    """[Updated version] - Basic partial tracker with no particular internal construct. Just dump your coordinates and do whatever you have to do with it!
    """
    def __init__(self):
        pass

    @staticmethod
    def generate_instance(x0, px0, y0, py0, cuda_device=None):
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
            return gpu_partial_track(x0, px0, y0, py0)
        else:
            return cpu_partial_track(x0, px0, y0, py0)


class cpu_partial_track(partial_track):
    def __init__(self, x0, px0, y0, py0, limit=100.0):

        # save data as members
        self.x0 = x0.copy()
        self.x  = x0.copy()
        self.px0 = px0.copy()
        self.px  = px0.copy()
        self.y0 = y0.copy()
        self.y  = y0.copy()
        self.py0 = py0.copy()
        self.py  = py0.copy()

        # For the modulation
        self.total_iters = 0
        self.step = np.zeros_like(self.x0)

        self.limit = limit

    def compute(self, n_iterations, epsilon, mu=0.0):
        """Compute the tracking
        
        Returns
        -------
        tuple of ndarray [n_elements]
            (radius, alpha, theta1, theta2, steps)
        """
        omega_x, omega_y = modulation(
            epsilon, n_iterations, self.total_iters)

        omega_x_cos = np.cos(omega_x)
        omega_x_sin = np.sin(omega_x)
        omega_y_cos = np.cos(omega_y)
        omega_y_sin = np.sin(omega_y)

        # Execution
        if mu == 0.0:
            self.x, self.px, self.y, self.py, self.step = cpu.henon_partial_track(
                self.x, self.px, self.y, self.py, self.step, self.limit, n_iterations, omega_x_sin, omega_x_cos, omega_y_cos, omega_y_sin
            )
        else:
            self.x, self.px, self.y, self.py, self.step = cpu.octo_henon_partial_track(
                self.x, self.px, self.y, self.py, self.step, self.limit, n_iterations, omega_x_sin, omega_x_cos, omega_y_cos, omega_y_sin, mu
            )

        self.total_iters += n_iterations
        return self.x, self.px, self.y, self.py, self.step

    def inverse_compute(self, n_iterations, epsilon):
        omega_x, omega_y = modulation(
            epsilon, n_iterations, self.total_iters, reversed=True)

        omega_x_cos = np.cos(omega_x)
        omega_x_sin = np.sin(omega_x)
        omega_y_cos = np.cos(omega_y)
        omega_y_sin = np.sin(omega_y)

        # Execution
        self.x, self.px, self.y, self.py, self.step = cpu.henon_inverse_partial_track(
            self.x, self.px, self.y, self.py, self.step, self.limit, n_iterations, omega_x_sin, omega_x_cos, omega_y_cos, omega_y_sin
        )

        self.total_iters -= n_iterations
        return self.x, self.px, self.y, self.py, self.step

    def reset(self):
        """Resets the engine
        """        
        self.x = self.x0.copy()
        self.px = self.px0.copy()
        self.y = self.y0.copy()
        self.py = self.py0.copy()
        self.total_iters = 0
        self.step = np.zeros_like(self.x0)

    def get_data(self):
        return self.x, self.px, self.y, self.py, self.step
    
    def get_zero_data(self):
        return self.x0, self.px0, self.y0, self.py0


class gpu_partial_track(partial_track):
    def __init__(self, x0, px0, y0, py0, limit=100.0):

        # save data as members
        self.x0 = x0.copy()
        self.x = x0.copy()
        self.d_x = cuda.to_device(x0)

        self.px0 = px0.copy()
        self.px = px0.copy()
        self.d_px = cuda.to_device(px0)
        
        self.y0 = y0.copy()
        self.y = y0.copy()
        self.d_y = cuda.to_device(y0)

        self.py0 = py0.copy()
        self.py = py0.copy()
        self.d_py = cuda.to_device(py0)

        # For the modulation
        self.total_iters = 0
        self.step = np.zeros_like(self.x0)
        self.d_step = cuda.to_device(self.step)

        self.limit = limit

    def compute(self, n_iterations, epsilon, mu=0.0):
        """Compute the tracking
        
        Returns
        -------
        tuple of ndarray [n_elements]
            (radius, alpha, theta1, theta2, steps)
        """
        threads_per_block = 512
        blocks_per_grid = self.x0.size // 512 + 1

        omega_x, omega_y = modulation(
            epsilon, n_iterations, self.total_iters)
        
        d_omega_x_sin = cuda.to_device(np.sin(omega_x))
        d_omega_x_cos = cuda.to_device(np.cos(omega_x))
        d_omega_y_sin = cuda.to_device(np.sin(omega_y))
        d_omega_y_cos = cuda.to_device(np.cos(omega_y))

        # Execution
        if mu == 0.0:
            gpu.henon_partial_track[blocks_per_grid, threads_per_block](
                self.d_x, self.d_px, self.d_y, self.d_py, self.d_step, self.limit,
                n_iterations, d_omega_x_sin, d_omega_x_cos, d_omega_y_sin, d_omega_y_cos
            )
        else:
            gpu.octo_henon_partial_track[blocks_per_grid, threads_per_block](
                self.d_x, self.d_px, self.d_y, self.d_py, self.d_step, self.limit,
                n_iterations, d_omega_x_sin, d_omega_x_cos, d_omega_y_sin, d_omega_y_cos, mu
            )
        self.total_iters += n_iterations

        self.d_x.copy_to_host(self.x)
        self.d_y.copy_to_host(self.y)
        self.d_px.copy_to_host(self.px)
        self.d_py.copy_to_host(self.py)
        self.d_step.copy_to_host(self.step)
        
        return self.x, self.px, self.y, self.py, self.step
        
    def inverse_compute(self, n_iterations, epsilon):
        threads_per_block = 512
        blocks_per_grid = self.x0.size // 512 + 1

        omega_x, omega_y = modulation(
            epsilon, n_iterations, self.total_iters, reversed=True)

        d_omega_x_cos = cuda.to_device(np.cos(omega_x))
        d_omega_x_sin = cuda.to_device(np.sin(omega_x))
        d_omega_y_cos = cuda.to_device(np.cos(omega_y))
        d_omega_y_sin = cuda.to_device(np.sin(omega_y))

        gpu.henon_inverse_partial_track[blocks_per_grid, threads_per_block](
            self.d_x, self.d_px, self.d_y, self.d_py, self.d_step, self.limit,
            n_iterations, d_omega_x_sin, d_omega_x_cos, d_omega_y_sin, d_omega_y_cos
        )
        self.total_iters -= n_iterations

        self.d_x.copy_to_host(self.x)
        self.d_y.copy_to_host(self.y)
        self.d_px.copy_to_host(self.px)
        self.d_py.copy_to_host(self.py)
        self.d_step.copy_to_host(self.step)

        return self.x, self.px, self.y, self.py, self.step

    def reset(self):
        """Resets the engine
        """        
        self.x = self.x0.copy()
        self.px = self.px0.copy()
        self.y = self.y0.copy()
        self.py = self.py0.copy()

        self.d_x = cuda.to_device(self.x0)
        self.d_px = cuda.to_device(self.px0)
        self.d_y = cuda.to_device(self.y0)
        self.d_py = cuda.to_device(self.py0)

        self.total_iters = 0
        self.step = np.zeros_like(self.x0)
        self.d_step = cuda.to_device(self.step)

    def get_data(self):
        return self.x, self.px, self.y, self.py, self.step

    def get_zero_data(self):
        return self.x0, self.px0, self.y0, self.py0


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
        omega_x_sin = np.sin(omega_x)
        omega_x_cos = np.cos(omega_x)
        omega_y_sin = np.sin(omega_y)
        omega_y_cos = np.cos(omega_y)

        d_omega_x_sin = cuda.to_device(
            np.asarray(omega_x_sin, dtype=np.float64))
        d_omega_x_cos = cuda.to_device(
            np.asarray(omega_x_cos, dtype=np.float64))
        d_omega_y_sin = cuda.to_device(
            np.asarray(omega_y_sin, dtype=np.float64))
        d_omega_y_cos = cuda.to_device(
            np.asarray(omega_y_cos, dtype=np.float64))

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
                d_x, d_px, d_y, d_py, d_times, 10.0, max_turns, d_omega_x_sin, d_omega_x_cos, d_omega_y_sin, d_omega_y_cos, np.float64(mu), d_bool_mask
            )

            d_times.copy_to_host(t_f)
            self.times[i] = t_f.reshape(x.shape)


class radial_block(object):
    """More advanced and complete radial_block analysis, replaces the deprecated radial_scan class"""
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

        self.r_list, self.dr = np.linspace(0, max_radius, radial_samples, retstep=True)
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
        omega_x_sin = np.sin(omega_x)
        omega_x_cos = np.cos(omega_x)
        omega_y_sin = np.sin(omega_y)
        omega_y_cos = np.cos(omega_y)

        d_omega_x_sin = cuda.to_device(
            np.asarray(omega_x_sin, dtype=np.float64))
        d_omega_x_cos = cuda.to_device(
            np.asarray(omega_x_cos, dtype=np.float64))
        d_omega_y_sin = cuda.to_device(
            np.asarray(omega_y_sin, dtype=np.float64))
        d_omega_y_cos = cuda.to_device(
            np.asarray(omega_y_cos, dtype=np.float64))

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
                    d_x, d_px, d_y, d_py, d_times, 10.0, max_turns, d_omega_x_sin, d_omega_x_cos, d_omega_y_sin, d_omega_y_cos, np.float64(mu), d_bool_mask
                )

                d_times.copy_to_host(t_f)
                self.times[i] = t_f.reshape(x.shape)


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

        # I HAVE A BUG THAT HAS TO BE FOUND
        self.times[0] = np.max(self.times)

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

        # COMPUTING BASELINE
        prelim_values = np.zeros(self.r_list.size)
        top_i = np.argmin(self.r_list <= cutting_point)
        if top_i == 0:
            top_i = len(self.weights)
        prelim_values[: top_i] = integrate.trapz(
            circ_trapz(
                circ_trapz(
                    self.weights[: top_i],
                    self.theta2,
                    np.pi * 2
                ),
                self.theta1,
                np.pi * 2
            ) * np.sin(self.alpha) * np.cos(self.alpha),
            self.alpha
        )
        baseline = integrate.trapz(prelim_values * np.power(self.r_list, 3), self.r_list)

        # COMPUTING THE REST
        values = np.empty(len(sample_list))
        for j, sample in enumerate(sample_list):
            prelim_values = np.zeros(self.r_list.size)
            ## THIS MAKES THE SAME CUTTING FOR THE WHOLE COMPUTATION ####
            top_i = np.argmin(self.r_list <= cutting_point)
            #############################################################
            if top_i == 0:
                top_i = len(self.weights)
            prelim_values[: top_i] = integrate.trapz(
                circ_trapz(
                    circ_trapz(
                        self.weights[: top_i] * (self.times[: top_i] >= sample),
                        self.theta2,
                        np.pi * 2
                    ),
                    self.theta1,
                    np.pi * 2
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
        prelim_values = np.zeros(self.r_list.size)

        for i, r in tqdm(enumerate(self.r_list), desc="integration...", total=len(self.r_list)):
            if r > cutting_point:
                prelim_values[i] = 0.0
            else:
                prelim_values[i] = integrate.trapz(
                    circ_trapz(
                        circ_trapz(
                            self.weights[i],
                            self.theta2,
                            np.pi * 2
                        ),
                        self.theta1,
                        np.pi * 2
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


def from_DA_to_loss_sym_gauss(sigma, DA_list, cut_point=None):
    loss_list = np.empty_like(DA_list)
    for i, DA in enumerate(DA_list):
        loss_list[i] = integrate.quad(
            lambda x: np.power(x, 3) * np.power(np.pi, 2) *
            2 * np.exp(- 0.5 * np.power(x / sigma, 2)),
            0.0, DA
        )[0]
    if not(cut_point is None):
        baseline = integrate.quad(
            lambda x: np.power(x, 3) * np.power(np.pi, 2) *
            2 * np.exp(- 0.5 * np.power(x / sigma, 2)),
            0.0, cut_point
        )[0]
        return loss_list / baseline
    else:
        return loss_list


def from_loss_to_DA_sym_gauss(sigma, loss_list, cut_point):
    baseline = integrate.quad(
        lambda x: np.power(x, 3) * np.power(np.pi, 2) *
        2 * np.exp(- 0.5 * np.power(x / sigma, 2)),
        0.0, cut_point
    )[0]
    def f(p):
        return integrate.quad(
            lambda x: np.power(x, 3) * np.power(np.pi, 2) *
            2 * np.exp(- 0.5 * np.power(x / sigma, 2)),
            0.0, p
        )[0] / baseline
    i_f = inversefunc(f, domain=((0.0, cut_point)))
    result = np.empty_like(loss_list)
    for i in range(len(loss_list)):
        result[i] = i_f(loss_list[i])
    return result
