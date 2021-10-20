"""The 4D Hénon map implemented follows the formulas presented in Eq.(15) in  
https://doi.org/10.1103/PhysRevE.53.4067
for the implementation of the sextupolar and octupolar kick.

You can have a modulation on the rotation coefficient regulated by epsilon and by the modulation function with the protocol displayed in
https://doi.org/10.1103/PhysRevE.57.3432

The octupolar kick is regulated by the parameter mu.
"""

from numba import cuda, jit, njit, prange
from numba.cuda.random import create_xoroshiro128p_states
import numpy as np

from . import gpu_henon_core as gpu
from . import cpu_henon_core as cpu

@njit
def modulation(epsilon, n_elements, first_index=0, reversed=False, kind="sps"):
    """Generates a modulation

    Parameters
    ----------
    epsilon : float
        intensity of modulation
    n_elements : float
        number of elements
    first_index : int, optional
        starting point of the modulation, by default 0
    kind : string, optional
        kind of modulation: "sps" or "simple" for now, by default "sps"
        * "sps" has the harmonics based from the article: https://doi.org/10.1103/PhysRevE.57.3432
        * "simple" has no harmonics just a single "omega" value that mixes with the two separate tunes

    Returns
    -------
    tuple of ndarray
        (omega_x, omega_y)
    """
    if "sps":
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
            number_list = list(
                range(first_index - n_elements, first_index))[::-1]

        omega_sum = np.array([
            np.sum(coefficients * np.cos(modulations * k)) for k in number_list
        ])
        omega_x = 0.168 * 2 * np.pi * (1 + epsilon * omega_sum)
        omega_y = 0.201 * 2 * np.pi * (1 + epsilon * omega_sum)
        return omega_x, omega_y
    elif "simple":
        omega = 2 * np.pi * 1.15e-3 * np.sqrt(2)
        if not reversed:
            number_list = list(range(first_index, first_index + n_elements))
        else:
            number_list = list(
                range(first_index - n_elements, first_index))[::-1]
        omega_x = 2 * np.pi * 0.201 * \
            (1 + epsilon * np.cos(omega * number_list))
        omega_y = 2 * np.pi * 0.168 * \
            (1 + epsilon * np.cos(omega * number_list))
        return omega_x, omega_y
    else:
        raise NotImplementedError


class partial_track(object):
    """[Updated version] - Basic partial tracker with no particular internal construct. Just dump your coordinates and do whatever you have to do with it!
    """

    def __init__(self):
        pass

    @staticmethod
    def generate_instance(x0, px0, y0, py0, limit=100.0, cuda_device=None):
        """Generate an instance of the engine.

        Parameters
        ----------
        radius : ndarray
            array of radiuses to consider
        x0 : ndarray
            array of initial x
        px0 : ndarray
            array of initial px
        y0 : ndarray
            array of initial y
        py0 : float
            array of initial py
        limit : float
            radial limit over which we consider the particle lost (default: 100.0)
        cuda_device : bool
            if set, forces the usage of GPU/CPU implementation (default: None)

        Returns
        -------
        class instance
            optimized class instance
        """
        if cuda_device == None:
            cuda_device = cuda.is_available()
        if cuda_device:
            return gpu_partial_track(x0, px0, y0, py0, limit)
        else:
            return cpu_partial_track(x0, px0, y0, py0, limit)


class cpu_partial_track(partial_track):
    def __init__(self, x0, px0, y0, py0, limit):
        """Init the class

        Parameters
        ----------
        x0 : ndarray
            initial condition
        px0 : ndarray
            initial condition
        y0 : ndarray
            initial condition
        py0 : ndarray
            initial condition
        limit : float
            radial barrier position
        """
        # save data as members
        self.x0 = x0.copy()
        self.x = x0.copy()
        self.px0 = px0.copy()
        self.px = px0.copy()
        self.y0 = y0.copy()
        self.y = y0.copy()
        self.py0 = py0.copy()
        self.py = py0.copy()

        # For the modulation
        self.total_iters = 0
        self.step = np.zeros_like(self.x0)

        self.limit = limit

    def compute(self, n_iterations, epsilon, mu=0.0, modulation_kind="sps", full_track=False, kick_module=None, kick_sigma=None):
        """Compute the traking with the given parameters and settings

        Parameters
        ----------
        n_iterations : int
            number of maximum iterations to perform
        epsilon : float
            modulation intensity
        mu : float, optional
            octupolar kick intensity, by default 0.0
        modulation_kind : str, optional
            kind of modulation, by default "sps"
        full_track : bool, optional
            do you want to know every single position of the particles along the tracking? By default False
        kick_module : float, optional
            Set module if you want a random kick at every iteration, by default None
        kick_sigma : float, optional
            Set sigma of kick distribution if you want a random kick at every iteration, by default None

        Returns
        -------
        tuple of ndarray [n_elements] if full_track == False
            (x, px, y, py, steps)
        tuple of ndarray [n_iterations, n_elements] if full_track == True
            (x, px, y, py)
        """
        omega_x, omega_y = modulation(
            epsilon, n_iterations, self.total_iters, kind=modulation_kind)

        omega_x_cos = np.cos(omega_x)
        omega_x_sin = np.sin(omega_x)
        omega_y_cos = np.cos(omega_y)
        omega_y_sin = np.sin(omega_y)

        if not full_track:
            # Execution
            if mu == 0.0:
                if kick_module is None or kick_sigma is None:
                    self.x, self.px, self.y, self.py, self.step = cpu.henon_partial_track(
                        self.x, self.px, self.y, self.py, self.step, self.limit, n_iterations, omega_x_sin, omega_x_cos, omega_y_sin, omega_y_cos
                    )
                else:
                    self.x, self.px, self.y, self.py, self.step = cpu.henon_partial_track_with_kick(
                        self.x, self.px, self.y, self.py, self.step, self.limit, n_iterations, omega_x_sin, omega_x_cos, omega_y_sin, omega_y_cos, kick_module, kick_sigma
                    )
            else:
                if kick_module is None or kick_sigma is None:
                    self.x, self.px, self.y, self.py, self.step = cpu.octo_henon_partial_track(
                        self.x, self.px, self.y, self.py, self.step, self.limit, n_iterations, omega_x_sin, omega_x_cos, omega_y_sin, omega_y_cos, mu
                    )
                else:
                    self.x, self.px, self.y, self.py, self.step = cpu.octo_henon_partial_track_with_kick(
                        self.x, self.px, self.y, self.py, self.step, self.limit, n_iterations, omega_x_sin, omega_x_cos, omega_y_sin, omega_y_cos, mu, kick_module, kick_sigma
                    )

            self.total_iters += n_iterations
            return self.x, self.px, self.y, self.py, self.step
        else:
            data_x = np.ones((n_iterations, self.x.size))
            data_px = np.ones((n_iterations, self.x.size))
            data_y = np.ones((n_iterations, self.x.size))
            data_py = np.ones((n_iterations, self.x.size))
            for i in range(n_iterations):
                # Execution
                if mu == 0.0:
                    if kick_module is None or kick_sigma is None:
                        self.x, self.px, self.y, self.py, self.step = cpu.henon_partial_track(
                            self.x, self.px, self.y, self.py, self.step, self.limit, 1, omega_x_sin[
                                i:i+1], omega_x_cos[i:i+1], omega_y_sin[i:i+1], omega_y_cos[i:i+1]
                        )
                    else:
                        self.x, self.px, self.y, self.py, self.step = cpu.henon_partial_track_with_kick(
                            self.x, self.px, self.y, self.py, self.step, self.limit, 1, omega_x_sin[
                                i:i+1], omega_x_cos[i:i+1], omega_y_sin[i:i+1], omega_y_cos[i:i+1], kick_module, kick_sigma
                        )
                else:
                    if kick_module is None or kick_sigma is None:
                        self.x, self.px, self.y, self.py, self.step = cpu.octo_henon_partial_track(
                            self.x, self.px, self.y, self.py, self.step, self.limit, 1, omega_x_sin[
                                i:i+1], omega_x_cos[i:i+1], omega_y_sin[i:i+1], omega_y_cos[i:i+1], mu
                        )
                    else:
                        self.x, self.px, self.y, self.py, self.step = cpu.octo_henon_partial_track_with_kick(
                            self.x, self.px, self.y, self.py, self.step, self.limit, 1, omega_x_sin[
                                i:i+1], omega_x_cos[i:i+1], omega_y_sin[i:i+1], omega_y_cos[i:i+1], mu, kick_module, kick_sigma
                        )
                data_x[i] = self.x
                data_px[i] = self.px
                data_y[i] = self.y
                data_py[i] = self.py
            self.total_iters += n_iterations
            return data_x, data_px, data_y, data_py

    def inverse_compute(self, n_iterations, epsilon, mu=0.0, modulation_kind="sps", kick_module=None, kick_sigma=None):
        """Compute an inverse tracking with the given parameters and settings

        Parameters
        ----------
        n_iterations : int
            number of maximum iterations to perform
        epsilon : float
            modulation intensity
        mu : float, optional
            octupolar kick intensity, by default 0.0
        modulation_kind : str, optional
            kind of modulation, by default "sps"
        full_track : bool, optional
            do you want to know every single position of the particles along the tracking? By default False
        kick_module : float, optional
            Set module if you want a random kick at every iteration, by default None
        kick_sigma : float, optional
            Set sigma of kick distribution if you want a random kick at every iteration, by default None

        Returns
        -------
        tuple of ndarray [n_elements]
            (x, px, y, py, steps)
        """
        omega_x, omega_y = modulation(
            epsilon, n_iterations, self.total_iters, reversed=True, kind=modulation_kind)

        omega_x_cos = np.cos(omega_x)
        omega_x_sin = np.sin(omega_x)
        omega_y_cos = np.cos(omega_y)
        omega_y_sin = np.sin(omega_y)

        # Execution
        if mu == 0.0:
            if kick_module is None or kick_sigma is None:
                self.x, self.px, self.y, self.py, self.step = cpu.henon_inverse_partial_track(
                    self.x, self.px, self.y, self.py, self.step, self.limit, n_iterations, omega_x_sin, omega_x_cos, omega_y_sin, omega_y_cos
                )
            else:
                self.x, self.px, self.y, self.py, self.step = cpu.henon_inverse_partial_track_with_kick(
                    self.x, self.px, self.y, self.py, self.step, self.limit, n_iterations, omega_x_sin, omega_x_cos, omega_y_sin, omega_y_cos, kick_module, kick_sigma
                )
        else:
            if kick_module is None or kick_sigma is None:
                self.x, self.px, self.y, self.py, self.step = cpu.octo_henon_inverse_partial_track(
                    self.x, self.px, self.y, self.py, self.step, self.limit, n_iterations, omega_x_sin, omega_x_cos, omega_y_sin, omega_y_cos, mu
                )
            else:
                self.x, self.px, self.y, self.py, self.step = cpu.octo_henon_inverse_partial_track_with_kick(
                    self.x, self.px, self.y, self.py, self.step, self.limit, n_iterations, omega_x_sin, omega_x_cos, omega_y_sin, omega_y_cos, mu, kick_module, kick_sigma
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
        """Get the particle positions and the step counter. 
        If the particle is lost, (x, px, y, py) are all set to NaN.

        Returns
        -------
        tuple of arrays
            x, px, y, py, steps
        """        
        return self.x, self.px, self.y, self.py, self.step

    def get_zero_data(self):
        """Get the initial conditions given to the engine

        Returns
        -------
        tuple of arrays
            x0, px0, y0, py0
        """        
        return self.x0, self.px0, self.y0, self.py0

    def add_kick(self, x=None, px=None, y=None, py=None):
        """Individually give a kick in cartesian coordinates to all the particles.

        Parameters
        ----------
        x : float or ndarray, optional
            give either a global or individual kick in x, by default None
        px : float or ndarray, optional
            give either a global or individual kick in px, by default None
        y : float or ndarray, optional
            give either a global or individual kick in y, by default None
        py : float or ndarray, optional
            give either a global or individual kick in py, by default None
        """        
        # Add kick
        if x is not None:
            self.x += x
        if px is not None:
            self.px += px
        if y is not None:
            self.y += y
        if py is not None:
            self.py += py


class gpu_partial_track(partial_track):
    def __init__(self, x0, px0, y0, py0, limit=100.0):
        """Init the class

        Parameters
        ----------
        x0 : ndarray
            initial condition
        px0 : ndarray
            initial condition
        y0 : ndarray
            initial condition
        py0 : ndarray
            initial condition
        limit : float
            radial barrier position
        """
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

    def compute(self, n_iterations, epsilon, mu=0.0, modulation_kind="sps", full_track=False, kick_module=None, kick_sigma=None):
        """Compute the traking with the given parameters and settings

        Parameters
        ----------
        n_iterations : int
            number of maximum iterations to perform
        epsilon : float
            modulation intensity
        mu : float, optional
            octupolar kick intensity, by default 0.0
        modulation_kind : str, optional
            kind of modulation, by default "sps"
        full_track : bool, optional
            do you want to know every single position of the particles along the tracking? By default False
        kick_module : float, optional
            Set module if you want a random kick at every iteration, by default None
        kick_sigma : float, optional
            Set sigma of kick distribution if you want a random kick at every iteration, by default None

        Returns
        -------
        tuple of ndarray [n_elements] if full_track == False
            (x, px, y, py, steps)
        tuple of ndarray [n_iterations, n_elements] if full_track == True
            (x, px, y, py)
        """
        threads_per_block = 512
        blocks_per_grid = self.x0.size // 512 + 1

        omega_x, omega_y = modulation(
            epsilon, n_iterations, self.total_iters, kind=modulation_kind)

        d_omega_x_sin = cuda.to_device(np.sin(omega_x))
        d_omega_x_cos = cuda.to_device(np.cos(omega_x))
        d_omega_y_sin = cuda.to_device(np.sin(omega_y))
        d_omega_y_cos = cuda.to_device(np.cos(omega_y))

        # Execution
        if not full_track:
            if mu == 0.0:
                if kick_module is None or kick_sigma is None:
                    gpu.henon_partial_track[blocks_per_grid, threads_per_block](
                        self.d_x, self.d_px, self.d_y, self.d_py, self.d_step, self.limit,
                        n_iterations, d_omega_x_sin, d_omega_x_cos, d_omega_y_sin, d_omega_y_cos
                    )
                else:
                    rng_states = create_xoroshiro128p_states(
                        threads_per_block * blocks_per_grid, seed=np.random.randint(0, 100000))
                    gpu.henon_partial_track_with_kick[blocks_per_grid, threads_per_block](
                        self.d_x, self.d_px, self.d_y, self.d_py, self.d_step, self.limit,
                        n_iterations, d_omega_x_sin, d_omega_x_cos, d_omega_y_sin, d_omega_y_cos, rng_states, kick_module, kick_sigma
                    )
            else:
                if kick_module is None or kick_sigma is None:
                    gpu.octo_henon_partial_track[blocks_per_grid, threads_per_block](
                        self.d_x, self.d_px, self.d_y, self.d_py, self.d_step, self.limit,
                        n_iterations, d_omega_x_sin, d_omega_x_cos, d_omega_y_sin, d_omega_y_cos, mu
                    )
                else:
                    rng_states = create_xoroshiro128p_states(
                        threads_per_block * blocks_per_grid, seed=np.random.randint(0, 100000))
                    gpu.octo_henon_partial_track_with_kick[blocks_per_grid, threads_per_block](
                        self.d_x, self.d_px, self.d_y, self.d_py, self.d_step, self.limit,
                        n_iterations, d_omega_x_sin, d_omega_x_cos, d_omega_y_sin, d_omega_y_cos, mu, rng_states, kick_module, kick_sigma
                    )
            self.total_iters += n_iterations

            self.d_x.copy_to_host(self.x)
            self.d_y.copy_to_host(self.y)
            self.d_px.copy_to_host(self.px)
            self.d_py.copy_to_host(self.py)
            self.d_step.copy_to_host(self.step)

            return self.x, self.px, self.y, self.py, self.step
        else:
            data_x = np.ones((n_iterations, self.x.size))
            data_px = np.ones((n_iterations, self.x.size))
            data_y = np.ones((n_iterations, self.x.size))
            data_py = np.ones((n_iterations, self.x.size))
            if kick_module is None or kick_sigma is None:
                for i in range(n_iterations):
                    if mu == 0.0:
                        gpu.henon_partial_track[blocks_per_grid, threads_per_block](
                            self.d_x, self.d_px, self.d_y, self.d_py, self.d_step, self.limit,
                            1, d_omega_x_sin[i:i+1], d_omega_x_cos[i:i +
                                                                   1], d_omega_y_sin[i:i+1], d_omega_y_cos[i:i+1]
                        )
                    else:
                        gpu.octo_henon_partial_track[blocks_per_grid, threads_per_block](
                            self.d_x, self.d_px, self.d_y, self.d_py, self.d_step, self.limit,
                            1, d_omega_x_sin[i:i+1], d_omega_x_cos[i:i +
                                                                   1], d_omega_y_sin[i:i+1], d_omega_y_cos[i:i+1], mu
                        )
                    self.d_x.copy_to_host(data_x[i])
                    self.d_y.copy_to_host(data_y[i])
                    self.d_px.copy_to_host(data_px[i])
                    self.d_py.copy_to_host(data_py[i])
            else:
                rng_states = create_xoroshiro128p_states(
                    threads_per_block * blocks_per_grid, seed=np.random.randint(0, 100000))
                for i in range(n_iterations):
                    if mu == 0.0:
                        gpu.henon_partial_track_with_kick[blocks_per_grid, threads_per_block](
                            self.d_x, self.d_px, self.d_y, self.d_py, self.d_step, self.limit,
                            1, d_omega_x_sin[i:i+1], d_omega_x_cos[i:i + 1], d_omega_y_sin[i:i+1], d_omega_y_cos[i:i+1], rng_states, kick_module, kick_sigma
                        )
                    else:
                        gpu.octo_henon_partial_track_with_kick[blocks_per_grid, threads_per_block](
                            self.d_x, self.d_px, self.d_y, self.d_py, self.d_step, self.limit,
                            1, d_omega_x_sin[i:i+1], d_omega_x_cos[i:i + 1], d_omega_y_sin[i:i+1], d_omega_y_cos[i:i+1], mu, rng_states, kick_module, kick_sigma
                        )
                    self.d_x.copy_to_host(data_x[i])
                    self.d_y.copy_to_host(data_y[i])
                    self.d_px.copy_to_host(data_px[i])
                    self.d_py.copy_to_host(data_py[i])

            self.total_iters += n_iterations
            return data_x, data_px, data_y, data_py

    def inverse_compute(self, n_iterations, epsilon, mu=0.0, modulation_kind="sps", kick_module=None, kick_sigma=None):
        """Compute an inverse tracking with the given parameters and settings

        Parameters
        ----------
        n_iterations : int
            number of maximum iterations to perform
        epsilon : float
            modulation intensity
        mu : float, optional
            octupolar kick intensity, by default 0.0
        modulation_kind : str, optional
            kind of modulation, by default "sps"
        full_track : bool, optional
            do you want to know every single position of the particles along the tracking? By default False
        kick_module : float, optional
            Set module if you want a random kick at every iteration, by default None
        kick_sigma : float, optional
            Set sigma of kick distribution if you want a random kick at every iteration, by default None

        Returns
        -------
        tuple of ndarray [n_elements]
            (x, px, y, py, steps)
        """
        threads_per_block = 512
        blocks_per_grid = self.x0.size // 512 + 1

        omega_x, omega_y = modulation(
            epsilon, n_iterations, self.total_iters, reversed=True, kind=modulation_kind)

        d_omega_x_cos = cuda.to_device(np.cos(omega_x))
        d_omega_x_sin = cuda.to_device(np.sin(omega_x))
        d_omega_y_cos = cuda.to_device(np.cos(omega_y))
        d_omega_y_sin = cuda.to_device(np.sin(omega_y))

        if kick_module is None or kick_sigma is None:
            if mu == 0.0:
                gpu.henon_inverse_partial_track[blocks_per_grid, threads_per_block](
                    self.d_x, self.d_px, self.d_y, self.d_py, self.d_step, self.limit,
                    n_iterations, d_omega_x_sin, d_omega_x_cos, d_omega_y_sin, d_omega_y_cos
                )
            else:
                gpu.octo_henon_inverse_partial_track[blocks_per_grid, threads_per_block](
                    self.d_x, self.d_px, self.d_y, self.d_py, self.d_step, self.limit,
                    n_iterations, d_omega_x_sin, d_omega_x_cos, d_omega_y_sin, d_omega_y_cos, mu
                )
        else:
            rng_states = create_xoroshiro128p_states(
                threads_per_block * blocks_per_grid, seed=np.random.randint(0, 100000))
            if mu == 0.0:
                gpu.henon_inverse_partial_track_with_kick[blocks_per_grid, threads_per_block](
                    self.d_x, self.d_px, self.d_y, self.d_py, self.d_step, self.limit,
                    n_iterations, d_omega_x_sin, d_omega_x_cos, d_omega_y_sin, d_omega_y_cos, rng_states, kick_module, kick_sigma
                )
            else:
                gpu.octo_henon_inverse_partial_track_with_kick[blocks_per_grid, threads_per_block](
                    self.d_x, self.d_px, self.d_y, self.d_py, self.d_step, self.limit,
                    n_iterations, d_omega_x_sin, d_omega_x_cos, d_omega_y_sin, d_omega_y_cos, mu, rng_states, kick_module, kick_sigma
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
        """Get the particle positions and the step counter. 
        If the particle is lost, (x, px, y, py) are all set to NaN.

        Returns
        -------
        tuple of arrays
            x, px, y, py, steps
        """
        return self.x, self.px, self.y, self.py, self.step

    def get_zero_data(self):
        """Get the initial conditions given to the engine

        Returns
        -------
        tuple of arrays
            x0, px0, y0, py0
        """
        return self.x0, self.px0, self.y0, self.py0

    def add_kick(self, x=None, px=None, y=None, py=None):
        """Individually give a kick in cartesian coordinates to all the particles.

        Parameters
        ----------
        x : float or ndarray, optional
            give either a global or individual kick in x, by default None
        px : float or ndarray, optional
            give either a global or individual kick in px, by default None
        y : float or ndarray, optional
            give either a global or individual kick in y, by default None
        py : float or ndarray, optional
            give either a global or individual kick in py, by default None
        """
        # Add kick
        if x is not None:
            self.x += x
        if px is not None:
            self.px += px
        if y is not None:
            self.y += y
        if py is not None:
            self.py += py
        # Update GPU
        self.d_x = cuda.to_device(self.x)
        self.d_px = cuda.to_device(self.px)
        self.d_y = cuda.to_device(self.y)
        self.d_py = cuda.to_device(self.py)
